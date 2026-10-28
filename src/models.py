"""Model zoo: every algorithm wrapped in a Pipeline with its own preprocessing.

Each model is a ``Pipeline([("prep", ColumnTransformer), ("clf", estimator)])``,
so scaling, log transforms and one-hot categories are fitted inside every
cross-validation training fold, never on validation or test rows.
"""

import io
import logging
from dataclasses import dataclass
from typing import Any

import joblib
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.features import FeatureGroups, build_scaled_preprocessor, build_tree_preprocessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one algorithm in the comparison."""

    key: str
    estimator: type[ClassifierMixin]
    preprocessing: str  # "scaled" (log + standardise) or "tree" (raw units)
    interpretability: str
    seeded: bool = True  # accepts random_state
    sample_size: int | None = None  # train on a stratified sample of this size

    @property
    def name(self) -> str:
        """Display name from ``config.MODEL_NAMES``."""
        return config.MODEL_NAMES[self.key]


MODEL_SPECS: dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        "dummy", DummyClassifier, "tree", "Baseline: always predicts 'not canceled'", seeded=False
    ),
    "logreg": ModelSpec(
        "logreg", LogisticRegression, "scaled", "High: one signed weight per feature"
    ),
    "tree": ModelSpec("tree", DecisionTreeClassifier, "tree", "High: readable if/else rules"),
    "rf": ModelSpec("rf", RandomForestClassifier, "tree", "Low: hundreds of deep trees"),
    "hgb": ModelSpec(
        "hgb",
        HistGradientBoostingClassifier,
        "tree",
        "Medium: additive trees, explained with permutation importance",
    ),
    "knn": ModelSpec(
        "knn",
        KNeighborsClassifier,
        "scaled",
        "Low: no global model, only similar past bookings",
        seeded=False,
        sample_size=config.KNN_SAMPLE_SIZE,
    ),
}


def base_key(key: str) -> str:
    """Strip the tuned suffix: ``"hgb_tuned"`` -> ``"hgb"``.

    Args:
        key: Model key, tuned or not.

    Returns:
        The key of the underlying algorithm.
    """
    return key.removesuffix(config.TUNED_SUFFIX)


def display_name(key: str) -> str:
    """Human-readable model name, with "(tuned)" for tuned variants.

    Args:
        key: Model key, tuned or not.

    Returns:
        Display name.
    """
    name = config.MODEL_NAMES[base_key(key)]
    return f"{name} (tuned)" if key.endswith(config.TUNED_SUFFIX) else name


def make_estimator(key: str, **overrides: Any) -> ClassifierMixin:  # noqa: ANN401
    """Create an unfitted estimator with the configured starting hyperparameters.

    Args:
        key: Algorithm key from ``MODEL_SPECS``.
        **overrides: Hyperparameters that replace the configured ones.

    Returns:
        The estimator.

    Raises:
        KeyError: If ``key`` is unknown.
    """
    spec = MODEL_SPECS[base_key(key)]
    params: dict[str, Any] = {**config.MODEL_PARAMS[spec.key], **overrides}
    if spec.seeded:
        params.setdefault("random_state", config.RANDOM_STATE)
    return spec.estimator(**params)


def build_pipeline(
    key: str, groups: FeatureGroups, params: dict[str, Any] | None = None
) -> Pipeline:
    """Wrap an estimator and its preprocessing into one Pipeline.

    Args:
        key: Algorithm key (``"_tuned"`` suffix allowed).
        groups: Model input columns.
        params: Estimator hyperparameters overriding the configured defaults
            (plain names, e.g. ``{"max_depth": 8}``).

    Returns:
        ``Pipeline([("prep", ...), ("clf", ...)])``.
    """
    spec = MODEL_SPECS[base_key(key)]
    builder = (
        build_scaled_preprocessor if spec.preprocessing == "scaled" else build_tree_preprocessor
    )
    return Pipeline([("prep", builder(groups)), ("clf", make_estimator(key, **(params or {})))])


def build_model_zoo(groups: FeatureGroups) -> dict[str, Pipeline]:
    """Build one untuned pipeline per algorithm (baseline first).

    Args:
        groups: Model input columns.

    Returns:
        Model key -> unfitted pipeline.
    """
    zoo = {key: build_pipeline(key, groups) for key in MODEL_SPECS}
    logger.info("Model zoo: %s", ", ".join(display_name(key) for key in zoo))
    return zoo


def model_size_mb(model: object) -> float:
    """Size of the model as a compressed joblib file, in megabytes.

    Args:
        model: Any picklable fitted model.

    Returns:
        Size in MB (same compression as the saved final model).
    """
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return round(len(buffer.getvalue()) / 1_000_000, 3)
