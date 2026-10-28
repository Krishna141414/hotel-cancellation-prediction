"""App data loading and the prediction service behind the Predict page.

The server only loads small precomputed JSON files and the final model, never
the dataset, so it fits comfortably in a 512 MB free instance.
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.build_cache import FORM_FIELDS, RANGE_FIELDS, SELECT_FIELDS, TOGGLE_FIELDS
from src import config
from src.features import FeatureGroups, engineer_features

logger = logging.getLogger(__name__)

# Sections of dashboard.json / explore.json served by /api/data/<section>.
DASHBOARD_SECTIONS: tuple[str, ...] = (
    "dataset",
    "setup",
    "models",
    "selection",
    "final",
    "threshold",
    "learning_curves",
    "validation_curve",
    "permutation",
    "tree_importance",
    "insights",
    "logreg_coefficients",
    "temporal",
    "ablation",
)
EXPLORE_SECTIONS: tuple[str, ...] = ("tree", "falling", "playground", "pca", "replay")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a cache file with a helpful error when it is missing.

    Args:
        path: JSON file.

    Returns:
        Parsed content.

    Raises:
        FileNotFoundError: If the file does not exist (run ``python main.py``).
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found - run `python main.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class AppData:
    """Everything the web app serves."""

    dashboard: dict[str, Any]
    explore: dict[str, Any]
    predict: dict[str, Any]
    model: Pipeline
    version: str = "dev"

    def section(self, name: str) -> Any:  # noqa: ANN401
        """One named section of the precomputed data.

        Args:
            name: Section name.

        Returns:
            The section's content.

        Raises:
            KeyError: If the section does not exist.
        """
        if name in DASHBOARD_SECTIONS:
            return self.dashboard[name]
        if name in EXPLORE_SECTIONS:
            return self.explore[name]
        if name == "predict":
            return self.predict
        raise KeyError(name)

    @property
    def final_model(self) -> dict[str, Any]:
        """The final model's record from the comparison."""
        key = self.dashboard["final"]["key"]
        return next(m for m in self.dashboard["models"] if m["key"] == key)


def cache_version(paths: tuple[Path, ...]) -> str:
    """Short content hash of the cache files, used to bust browser caches after a redeploy.

    Args:
        paths: Cache files.

    Returns:
        First 10 hex characters of their combined SHA-256.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def load_app_data() -> AppData:
    """Load the caches and the final model from disk.

    Returns:
        The app data.
    """
    cache_files = (config.APP_DASHBOARD_CACHE, config.APP_EXPLORE_CACHE, config.APP_PREDICT_CACHE)
    data = AppData(
        dashboard=_read_json(config.APP_DASHBOARD_CACHE),
        explore=_read_json(config.APP_EXPLORE_CACHE),
        predict=_read_json(config.APP_PREDICT_CACHE),
        model=joblib.load(config.FINAL_MODEL_PATH),
        version=cache_version(cache_files),
    )
    logger.info("Loaded app data and final model (%s)", data.dashboard["final"]["name"])
    return data


# ---------------------------------------------------------------------------
# Input validation and conversion
# ---------------------------------------------------------------------------
def validate_booking(payload: dict[str, Any], schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Check and coerce one booking sent by the Predict form.

    Args:
        payload: Raw JSON body.
        schema: Form schema from ``predict.json``.

    Returns:
        A clean booking (every field present, correct types).

    Raises:
        ValueError: With a user-friendly message if anything is missing or invalid.
    """
    if not isinstance(payload, dict):
        raise ValueError("Send the booking as a JSON object.")
    missing = [name for name in FORM_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    by_name = {field["name"]: field for field in schema}
    booking: dict[str, Any] = {}
    for name in RANGE_FIELDS:
        field = by_name[name]
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            raise ValueError(f"{field['label']} must be a number.") from None
        if not field["min"] <= value <= field["max"] or not np.isfinite(value):
            raise ValueError(f"{field['label']} must be between {field['min']} and {field['max']}.")
        booking[name] = int(round(value)) if name != "adr" else round(value, 2)
    for name in SELECT_FIELDS:
        allowed = {str(option["value"]): option["value"] for option in by_name[name]["options"]}
        if str(payload[name]) not in allowed:
            raise ValueError(f"Unknown value for {by_name[name]['label']}: {payload[name]!r}")
        booking[name] = allowed[str(payload[name])]
    for name in TOGGLE_FIELDS:
        value = payload[name]
        if value not in (0, 1, True, False, "0", "1"):
            raise ValueError(f"{by_name[name]['label']} must be 0 or 1.")
        booking[name] = int(value in (1, True, "1"))
    try:
        booking["arrival_date"] = date.fromisoformat(str(payload["arrival_date"])).isoformat()
    except ValueError:
        raise ValueError("Arrival date must look like YYYY-MM-DD.") from None
    if booking["adults"] + booking["children"] == 0:
        raise ValueError("A booking needs at least one adult or child.")
    return booking


def bookings_to_inputs(bookings: list[dict[str, Any]], groups: FeatureGroups) -> pd.DataFrame:
    """Turn form bookings into model inputs (same feature engineering as training).

    Args:
        bookings: Validated bookings.
        groups: Model input columns.

    Returns:
        One row per booking with exactly ``groups.all`` columns.
    """
    frame = pd.DataFrame(bookings)
    dates = pd.to_datetime(frame.pop("arrival_date"))
    frame["arrival_date_year"] = dates.dt.year
    frame["arrival_date_month"] = dates.dt.month_name()
    frame["arrival_date_day_of_month"] = dates.dt.day
    frame["has_agent"] = (frame["agent"] != config.NO_AGENT_ID).astype(int)
    return engineer_features(frame)[groups.all]


def _same_value(a: object, b: object) -> bool:
    """Compare form values numerically when possible (``99`` == ``99.0``), else as text.

    Args:
        a: First value.
        b: Second value.

    Returns:
        Whether the values are equal.
    """
    try:
        return float(a) == float(b)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(a) == str(b)


# ---------------------------------------------------------------------------
# Prediction service
# ---------------------------------------------------------------------------
class Predictor:
    """Probability, explanation and what-if curves for one booking.

    Args:
        data: Loaded app data.
    """

    def __init__(self, data: AppData) -> None:
        """Keep references to the model, schema and reference booking."""
        self.model = data.model
        self.schema = data.predict["fields"]
        self.defaults = data.predict["defaults"]
        self.threshold = float(data.predict["threshold"])
        self.groups = FeatureGroups(
            numeric=data.dashboard["dataset"]["numeric"],
            binary=data.dashboard["dataset"]["binary"],
            categorical=data.dashboard["dataset"]["categorical"],
        )
        self.labels = {field["name"]: field["label"] for field in self.schema}
        self.option_labels = {
            field["name"]: {str(o["value"]): o["label"] for o in field["options"]}
            for field in self.schema
            if field["widget"] == "select"
        }

    def _value_label(self, field: str, value: object) -> str:
        """Readable value for the explanation list.

        Args:
            field: Field name.
            value: Field value.

        Returns:
            Display text.
        """
        if field in self.option_labels:
            return self.option_labels[field].get(str(value), str(value))
        if field in TOGGLE_FIELDS:
            return "yes" if value else "no"
        if field == "arrival_date":
            return pd.Timestamp(value).strftime("%d %b")
        return f"{value:g}" if isinstance(value, float) else str(value)

    def _variants(self, booking: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        """Copies of the booking with one field reset to the typical value.

        Args:
            booking: Validated booking.

        Returns:
            ``(variants, changed_fields)``.
        """
        variants, fields = [], []
        for field in FORM_FIELDS:
            reference = self.defaults[field]
            if field == "arrival_date":
                # Keep the year; only month/day carry signal.
                reference = f"{booking[field][:4]}{reference[4:]}"
            if _same_value(booking[field], reference):
                continue
            variant = dict(booking)
            variant[field] = reference
            if variant["adults"] + variant["children"] == 0:
                continue
            variants.append(variant)
            fields.append(field)
        return variants, fields

    def _sweeps(self, booking: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list]]:
        """What-if rows: the booking with lead time / ADR swept over a range.

        Args:
            booking: Validated booking.

        Returns:
            ``(rows, {field: x_values})``.
        """
        rows, xs = [], {}
        for field, (start, stop) in config.SWEEP_RANGES.items():
            values = np.linspace(start, stop, config.SWEEP_POINTS).round().tolist()
            xs[field] = values
            rows.extend({**booking, field: value} for value in values)
        return rows, xs

    def risk_band(self, probability: float) -> str:
        """Low / medium / high relative to the decision threshold.

        Args:
            probability: Predicted probability.

        Returns:
            Band name.
        """
        if probability >= self.threshold:
            return "high"
        return "low" if probability < config.RISK_LOW else "medium"

    def predict(self, booking: dict[str, Any]) -> dict[str, Any]:
        """Predict, explain and sweep one validated booking (one batched model call).

        "Why this prediction": each field's contribution is how much the probability
        changes when only that field is reset to the typical booking's value.

        Args:
            booking: Validated booking.

        Returns:
            Probability, verdict, risk band, contributions and what-if curves.
        """
        variants, fields = self._variants(booking)
        sweep_rows, sweep_x = self._sweeps(booking)
        rows = [booking, *variants, *sweep_rows]
        proba = self.model.predict_proba(bookings_to_inputs(rows, self.groups))[:, 1]
        probability = float(proba[0])
        variant_proba = proba[1 : 1 + len(variants)]
        contributions = [
            {
                "field": field,
                "label": self.labels[field],
                "value": self._value_label(field, booking[field]),
                "typical": self._value_label(field, variant[field]),
                "delta": round(probability - float(p), 4),
            }
            for field, variant, p in zip(fields, variants, variant_proba, strict=True)
        ]
        contributions.sort(key=lambda item: abs(item["delta"]), reverse=True)
        sweeps, offset = {}, 1 + len(variants)
        for field, values in sweep_x.items():
            sweeps[field] = {
                "x": values,
                "y": np.round(proba[offset : offset + len(values)], 4).tolist(),
                "current": booking[field],
            }
            offset += len(values)
        return {
            "probability": round(probability, 4),
            "threshold": self.threshold,
            "prediction": int(probability >= self.threshold),
            "risk": self.risk_band(probability),
            "contributions": contributions,
            "sweeps": sweeps,
        }


def random_sample(data: AppData, rng: random.Random | None = None) -> dict[str, Any]:
    """One random real test booking (with its true outcome) for the Predict form.

    Args:
        data: App data.
        rng: Random generator (tests pass a seeded one).

    Returns:
        A booking in form format plus ``"actual"``.
    """
    return (rng or random).choice(data.predict["samples"])
