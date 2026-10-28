"""Tests for the feature-engineering and selection functions in src.features (from Task 2)."""

import pandas as pd
import pytest

from src import config, features


@pytest.fixture
def clean_bookings() -> pd.DataFrame:
    """Three cleaned bookings with known, hand-checkable feature values."""
    return pd.DataFrame(
        {
            "stays_in_weekend_nights": [2, 0, 1],
            "stays_in_week_nights": [3, 0, 1],
            "adults": [2, 1, 2],
            "children": [1, 0, 0],
            "babies": [0, 0, 1],
            "adr": [90.0, 0.0, 100.0],
            "arrival_date_year": [2016, 2017, 2015],
            "arrival_date_month": ["July", "January", "December"],
            "arrival_date_day_of_month": [4, 1, 25],
            "country": ["PRT", "GBR", "Unknown"],
            "previous_cancellations": [0, 1, 3],
            "previous_bookings_not_canceled": [0, 3, 0],
        }
    )


def test_add_stay_features(clean_bookings: pd.DataFrame) -> None:
    """Nights and guests are summed; babies count as children for the flag."""
    result = features.add_stay_features(clean_bookings)
    assert result["total_nights"].tolist() == [5, 0, 2]
    assert result["total_guests"].tolist() == [3, 1, 3]
    assert result["has_children"].tolist() == [1, 0, 1]


def test_add_price_features(clean_bookings: pd.DataFrame) -> None:
    """Per-person price excludes babies; stay value = ADR x nights."""
    result = features.add_price_features(features.add_stay_features(clean_bookings))
    assert result["adr_per_person"].tolist() == [30.0, 0.0, 50.0]
    assert result["stay_value"].tolist() == [450.0, 0.0, 200.0]


def test_add_price_features_rejects_zero_paying_guests() -> None:
    """A booking with only babies would divide by zero and is rejected."""
    df = pd.DataFrame({"adr": [50.0], "adults": [0], "children": [0], "total_nights": [1]})
    with pytest.raises(ValueError, match="paying guests"):
        features.add_price_features(df)


def test_add_arrival_features(clean_bookings: pd.DataFrame) -> None:
    """Month number, weekday and season are derived from the split date columns."""
    result = features.add_arrival_features(clean_bookings)
    assert result["arrival_month_num"].tolist() == [7, 1, 12]
    # 2016-07-04 Monday, 2017-01-01 Sunday, 2015-12-25 Friday
    assert result["arrival_weekday"].tolist() == [0, 6, 4]
    assert result["arrival_season"].tolist() == ["Summer", "Winter", "Winter"]


def test_add_arrival_features_rejects_bad_month(clean_bookings: pd.DataFrame) -> None:
    """Unknown month names raise a clear error."""
    bad = clean_bookings.assign(arrival_date_month=["Jul", "January", "December"])
    with pytest.raises(ValueError, match="Jul"):
        features.add_arrival_features(bad)


def test_add_guest_history_features(clean_bookings: pd.DataFrame) -> None:
    """Domestic flag uses PRT; cancel ratio is 0 for guests with no history."""
    result = features.add_guest_history_features(clean_bookings)
    assert result["is_domestic"].tolist() == [1, 0, 0]
    assert result["previous_cancel_ratio"].tolist() == [0.0, 0.25, 1.0]


def test_engineer_features_adds_all_configured_columns(clean_bookings: pd.DataFrame) -> None:
    """The full run adds every engineered feature and keeps the row count."""
    result = features.engineer_features(clean_bookings)
    assert set(config.ENGINEERED_FEATURES) <= set(result.columns)
    assert len(result) == len(clean_bookings)


def test_engineer_features_requires_inputs() -> None:
    """Missing input columns raise a clear error."""
    with pytest.raises(ValueError, match="Missing columns"):
        features.engineer_features(pd.DataFrame({"adr": [1.0]}))


# ---------------------------------------------------------------------------
# Feature selection checks
# ---------------------------------------------------------------------------
@pytest.fixture
def train_rows() -> pd.DataFrame:
    """Ten training rows: one leaky column, one redundant pair, one near-constant column."""
    return pd.DataFrame(
        {
            "is_canceled": [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
            "status": ["C", "C", "C", "OK", "OK", "OK", "OK", "C", "OK", "OK"],
            "hotel": ["A", "B"] * 5,
            "nights": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "nights_copy": [2, 4, 6, 8, 10, 12, 14, 16, 18, 21],
            "lead_time": [5, 1, 9, 2, 8, 3, 7, 4, 6, 0],
            "babies": [0] * 9 + [1],
        }
    )


def test_target_purity(train_rows: pd.DataFrame) -> None:
    """A leaky column scores 1.0; an uninformative one scores the majority share."""
    assert features.target_purity(train_rows, "status") == 1.0
    assert features.target_purity(train_rows, "hotel") == pytest.approx(0.6)


def test_find_leakage_columns(train_rows: pd.DataFrame) -> None:
    """Only the column that perfectly predicts the target is flagged."""
    assert features.find_leakage_columns(train_rows, ["status", "hotel"]) == {"status": 1.0}


def test_find_correlated_pairs(train_rows: pd.DataFrame) -> None:
    """The near-copy is reported; unrelated columns are not."""
    pairs = features.find_correlated_pairs(train_rows, ["nights", "nights_copy", "lead_time"])
    assert pairs[["feature_a", "feature_b"]].values.tolist() == [["nights", "nights_copy"]]


def test_find_near_constant(train_rows: pd.DataFrame) -> None:
    """A column that is 90% one value is flagged at a 0.9 threshold."""
    flagged = features.find_near_constant(train_rows, ["babies", "nights"], threshold=0.9)
    assert flagged == {"babies": 0.9}


def test_validate_selection_rejects_leaky_feature(train_rows: pd.DataFrame) -> None:
    """A final feature set that still contains leakage fails loudly."""
    with pytest.raises(ValueError, match="selection check failed"):
        features.validate_selection(train_rows, ["nights", "lead_time"], ["status"])


def test_validate_selection_rejects_redundant_pair(train_rows: pd.DataFrame) -> None:
    """A final feature set with a redundant pair fails loudly."""
    with pytest.raises(ValueError, match="redundant pairs"):
        features.validate_selection(train_rows, ["nights", "nights_copy"], ["hotel"])


def test_validate_selection_accepts_clean_set(train_rows: pd.DataFrame) -> None:
    """A clean final set passes without error."""
    features.validate_selection(train_rows, ["nights", "lead_time"], ["hotel"])


def test_configured_drops_are_candidates() -> None:
    """Every dropped feature is a real candidate (catches typos in config)."""
    candidates = set(config.NUMERIC_FEATURES + config.BINARY_FEATURES + config.CATEGORICAL_FEATURES)
    assert set(config.DROPPED_FEATURES) <= candidates
