"""Build ``app/cache/predict.json``: the Predict form schema, defaults and sample bookings.

Run on its own with ``python -m app.build_cache`` (``main.py`` also calls it).
Everything comes from the training rows (options, typical values) or the test
rows (random sample bookings), so the server never needs the dataset.
"""

import json
import logging
from typing import Any

import pandas as pd

from src import config
from src.data import clean_data, load_raw_data
from src.features import MONTH_NAME_TO_NUMBER, arrival_dates, engineer_features, split_frame

logger = logging.getLogger(__name__)

# Field name -> (label, group, widget, help text). Order = order on the page.
FORM_FIELDS: dict[str, tuple[str, str, str, str]] = {
    "hotel": ("Hotel", "Stay", "select", "City hotel in Lisbon or resort in the Algarve"),
    "arrival_date": ("Arrival date", "Stay", "date", "Gives month, season and weekday"),
    "stays_in_weekend_nights": ("Weekend nights", "Stay", "range", "Saturday/Sunday nights"),
    "stays_in_week_nights": ("Week nights", "Stay", "range", "Monday-Friday nights"),
    "adults": ("Adults", "Stay", "range", ""),
    "children": ("Children", "Stay", "range", ""),
    "babies": ("Babies", "Stay", "range", ""),
    "lead_time": ("Lead time (days)", "Booking", "range", "Days between booking and arrival"),
    "adr": ("Daily rate (EUR)", "Booking", "range", "Average daily rate"),
    "deposit_type": ("Deposit", "Booking", "select", ""),
    "market_segment": ("Market segment", "Booking", "select", ""),
    "distribution_channel": ("Channel", "Booking", "select", ""),
    "customer_type": ("Customer type", "Booking", "select", ""),
    "meal": ("Meal plan", "Booking", "select", ""),
    "reserved_room_type": ("Room type", "Booking", "select", "Anonymised room category"),
    "agent": ("Travel agent", "Booking", "select", "Anonymised agent ID"),
    "has_company": ("Paid by a company", "Booking", "toggle", ""),
    "days_in_waiting_list": ("Days on waiting list", "Booking", "range", ""),
    "country": ("Country", "Guest", "select", "Guest nationality"),
    "is_repeated_guest": ("Repeat guest", "Guest", "toggle", "Stayed at this hotel before"),
    "previous_cancellations": ("Previous cancellations", "Guest", "range", ""),
    "previous_bookings_not_canceled": ("Previous stays kept", "Guest", "range", ""),
    "total_of_special_requests": ("Special requests", "Guest", "range", "e.g. high floor, twin"),
    "required_car_parking_spaces": ("Parking spaces", "Guest", "range", ""),
}
SELECT_FIELDS: list[str] = [name for name, spec in FORM_FIELDS.items() if spec[2] == "select"]
TOGGLE_FIELDS: list[str] = [name for name, spec in FORM_FIELDS.items() if spec[2] == "toggle"]
RANGE_FIELDS: list[str] = [name for name, spec in FORM_FIELDS.items() if spec[2] == "range"]

VALUE_LABELS: dict[str, dict[str, str]] = {
    "meal": {"BB": "Bed & breakfast", "HB": "Half board", "FB": "Full board", "SC": "No meal"},
    "country": {
        "PRT": "Portugal", "GBR": "United Kingdom", "FRA": "France", "ESP": "Spain",
        "DEU": "Germany", "ITA": "Italy", "IRL": "Ireland", "BEL": "Belgium", "BRA": "Brazil",
        "NLD": "Netherlands", "USA": "United States", "CHE": "Switzerland", "AUT": "Austria",
        "CN": "China", "SWE": "Sweden", "POL": "Poland", config.UNKNOWN_COUNTRY: "Unknown",
    },
}  # fmt: skip


def _option_label(field: str, value: object) -> str:
    """Readable label for a select option.

    Args:
        field: Field name.
        value: Option value.

    Returns:
        Label text.
    """
    if field == "agent":
        if value == config.NO_AGENT_ID:
            return "No agent (direct)"
        return "Other agent" if value == config.OTHER_AGENT_ID else f"Agent {value}"
    if field == "reserved_room_type":
        return f"Room type {value}"
    return VALUE_LABELS.get(field, {}).get(str(value), str(value))


def _select_options(train_rows: pd.DataFrame, field: str) -> list[dict[str, Any]]:
    """Options for one select box, most common first.

    Args:
        train_rows: Training bookings.
        field: Categorical field.

    Returns:
        ``[{"value", "label"}]``.
    """
    counts = train_rows[field].value_counts()
    if field == "country":
        values = list(counts.index[: config.FORM_TOP_COUNTRIES]) + [config.OTHER_OPTION]
    elif field == "agent":
        top = [v for v in counts.index if v != config.NO_AGENT_ID][: config.FORM_TOP_AGENTS]
        values = [config.NO_AGENT_ID, *top, config.OTHER_AGENT_ID]
    else:
        values = list(counts[counts >= config.FORM_MIN_CATEGORY_ROWS].index)
    if field == "reserved_room_type":
        values = sorted(values)
    return [{"value": _plain(v), "label": _option_label(field, v)} for v in values]


def _plain(value: object) -> object:
    """Convert numpy scalars to plain Python values.

    Args:
        value: Any scalar.

    Returns:
        A JSON-friendly scalar.
    """
    return value.item() if hasattr(value, "item") else value


def typical_booking(train_rows: pd.DataFrame) -> dict[str, Any]:
    """The 'typical' booking: median of numeric fields, most common category otherwise.

    Used as the form's starting point and as the reference for "why this prediction".

    Args:
        train_rows: Training bookings.

    Returns:
        Field -> value.
    """
    booking: dict[str, Any] = {}
    for field in FORM_FIELDS:
        if field == "arrival_date":
            month = train_rows["arrival_date_month"].mode().iloc[0]
            year = int(train_rows["arrival_date_year"].max())
            booking[field] = f"{year}-{MONTH_NAME_TO_NUMBER[month]:02d}-15"
        elif field in RANGE_FIELDS:
            booking[field] = int(round(float(train_rows[field].median())))
        else:
            booking[field] = _plain(train_rows[field].mode().iloc[0])
    for field in TOGGLE_FIELDS:
        booking[field] = int(booking[field])
    return booking


def row_to_booking(row: pd.Series, date: pd.Timestamp) -> dict[str, Any]:
    """Convert one cleaned booking to the form's field format.

    Args:
        row: Cleaned + engineered booking.
        date: Its arrival date.

    Returns:
        Field -> value.
    """
    booking = {field: _plain(row[field]) for field in FORM_FIELDS if field != "arrival_date"}
    booking["arrival_date"] = date.strftime("%Y-%m-%d")
    return booking


def build_schema(train_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Describe every form field (widget, label, options or range).

    Args:
        train_rows: Training bookings.

    Returns:
        One dictionary per field, in page order.
    """
    fields = []
    for name, (label, group, widget, help_text) in FORM_FIELDS.items():
        field: dict[str, Any] = {"name": name, "label": label, "group": group, "widget": widget}
        if help_text:
            field["help"] = help_text
        if widget == "select":
            field["options"] = _select_options(train_rows, name)
        elif widget == "range":
            low, high, stride = config.NUMERIC_INPUT_RANGES[name]
            field |= {"min": low, "max": high, "step": stride}
        fields.append(field)
    return fields


def _snap_to_options(booking: dict[str, Any], schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Map category values that are not offered in the form to "Other".

    Args:
        booking: Booking in form format.
        schema: Form schema.

    Returns:
        Booking whose select values all exist in the form.
    """
    for field in schema:
        if field["widget"] != "select":
            continue
        values = [option["value"] for option in field["options"]]
        if booking[field["name"]] not in values:
            fallback = config.OTHER_AGENT_ID if field["name"] == "agent" else config.OTHER_OPTION
            booking[field["name"]] = fallback if fallback in values else values[0]
    for field in schema:
        if field["widget"] == "range":
            booking[field["name"]] = min(max(booking[field["name"]], field["min"]), field["max"])
    return booking


def sample_bookings(
    test_rows: pd.DataFrame, schema: list[dict[str, Any]], size: int = config.APP_RANDOM_SAMPLES
) -> list[dict[str, Any]]:
    """Random real test bookings for the "try a random booking" button.

    Args:
        test_rows: Test bookings.
        schema: Form schema (values outside the form's options are snapped).
        size: Number of samples.

    Returns:
        Bookings in form format with their true outcome in ``"actual"``.
    """
    sample = test_rows.sample(size, random_state=config.RANDOM_STATE)
    dates = arrival_dates(sample)
    bookings = []
    for (_, row), date in zip(sample.iterrows(), dates, strict=True):
        booking = _snap_to_options(row_to_booking(row, date), schema)
        booking["actual"] = int(row[config.TARGET])
        bookings.append(booking)
    return bookings


def build_predict_cache(featured: pd.DataFrame | None = None) -> dict[str, Any]:
    """Build and save ``app/cache/predict.json``.

    Args:
        featured: Cleaned + engineered bookings; rebuilt from the raw CSV if ``None``.

    Returns:
        The cache payload.
    """
    if featured is None:
        featured = engineer_features(clean_data(load_raw_data()).data)
    train_rows, test_rows = split_frame(featured)
    schema = build_schema(train_rows)
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    payload = {
        "fields": schema,
        "defaults": _snap_to_options(typical_booking(train_rows), schema),
        "samples": sample_bookings(test_rows, schema),
        "threshold": metrics["final"]["threshold"],
        "model_name": metrics["final"]["name"],
        "risk_low": config.RISK_LOW,
    }
    config.APP_PREDICT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    config.APP_PREDICT_CACHE.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info(
        "Saved %s (%d fields, %d sample bookings)",
        config.APP_PREDICT_CACHE.relative_to(config.PROJECT_ROOT),
        len(schema),
        len(payload["samples"]),
    )
    return payload


if __name__ == "__main__":
    config.setup_logging()
    build_predict_cache()
