"""Tests for the splits, feature groups and preprocessing transformers in src.features."""

import numpy as np
import pandas as pd
import pytest

from src import config, features
from src.features import FeatureGroups

GROUPS = FeatureGroups(
    numeric=["lead_time", "adr"],
    binary=["has_agent"],
    categorical=["hotel", "country", "agent"],
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """400 synthetic bookings with a skewed column and a rare country (3 rows)."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = 400
    countries = rng.choice(["PRT", "GBR"], n)
    countries[:3] = "ZZZ"
    return pd.DataFrame(
        {
            "lead_time": rng.exponential(60, n).round(),
            "adr": rng.normal(100, 20, n).round(2),
            "has_agent": rng.integers(0, 2, n),
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
            "country": countries,
            "agent": rng.choice([0, 9, 240], n),
            "is_canceled": rng.integers(0, 2, n),
        }
    )


def test_selected_groups_exclude_dropped_and_leaky_columns() -> None:
    """The final inputs never contain a dropped or outcome column."""
    selected = set(FeatureGroups.selected().all)
    assert not selected & set(config.DROPPED_FEATURES)
    assert not selected & set(config.LEAKAGE_COLUMNS)
    assert len(selected) == 33


def test_with_columns_adds_to_the_right_group() -> None:
    """Candidate columns go back into their own group; unknown names are rejected."""
    extended = GROUPS.with_columns(["booking_changes", "assigned_room_type"])
    assert "booking_changes" in extended.numeric
    assert "assigned_room_type" in extended.categorical
    with pytest.raises(ValueError, match="Not candidate"):
        GROUPS.with_columns(["reservation_status"])


def test_split_frame_is_stratified(frame: pd.DataFrame) -> None:
    """80/20 split with the same cancel rate and no shared rows."""
    train, test = features.split_frame(frame)
    assert len(train) == 320 and len(test) == 80
    assert set(train.index).isdisjoint(test.index)
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.02


def test_temporal_split_orders_by_arrival(featured_sample: pd.DataFrame) -> None:
    """Every test arrival is on/after the cutoff, every train arrival before it."""
    train, test, cutoff = features.temporal_split(featured_sample, test_share=0.2)
    assert features.arrival_dates(train).max() < cutoff <= features.arrival_dates(test).min()
    assert 0.1 < len(test) / len(featured_sample) < 0.3


def test_temporal_split_rejects_bad_share(featured_sample: pd.DataFrame) -> None:
    """Shares outside (0, 1) are rejected."""
    with pytest.raises(ValueError, match="test_share"):
        features.temporal_split(featured_sample, test_share=1.0)


def test_skew_log1p_learns_columns_on_fit_data_only(frame: pd.DataFrame) -> None:
    """The skewed column is chosen during fit and transformed; others are untouched."""
    transformer = features.SkewLog1p().fit(frame[["lead_time", "adr"]])
    assert transformer.log_columns_ == ["lead_time"]
    out = transformer.transform(frame[["lead_time", "adr"]])
    np.testing.assert_allclose(out["lead_time"], np.log1p(frame["lead_time"]))
    pd.testing.assert_series_equal(out["adr"], frame["adr"])


def test_scaled_preprocessor_is_fitted_on_train_only(frame: pd.DataFrame) -> None:
    """Scaler means equal the training-rows statistics, not the full data's."""
    train, _ = features.split_frame(frame)
    preprocessor = features.build_scaled_preprocessor(GROUPS).fit(train[GROUPS.all])
    scaler = preprocessor.named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[1] == pytest.approx(train["adr"].mean())
    assert scaler.mean_[1] != pytest.approx(frame["adr"].mean())


def test_tree_preprocessor_keeps_raw_units(frame: pd.DataFrame) -> None:
    """Tree preprocessing does not scale numeric columns (readable thresholds)."""
    out = features.build_tree_preprocessor(GROUPS).fit_transform(frame[GROUPS.all])
    pd.testing.assert_series_equal(
        out["num__lead_time"], frame["lead_time"], check_names=False, check_dtype=False
    )


def test_rare_and_unknown_categories_go_to_infrequent(frame: pd.DataFrame) -> None:
    """Rare categories are merged; unseen ones map to 'infrequent' instead of crashing."""
    preprocessor = features.build_tree_preprocessor(GROUPS)
    output = preprocessor.fit_transform(frame[GROUPS.all])
    assert "cat__country_ZZZ" not in output.columns
    unseen = preprocessor.transform(frame.head(1)[GROUPS.all].assign(country="NEW"))
    assert unseen["cat__country_infrequent_sklearn"].iloc[0] == 1.0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("num__lead_time", ("lead_time", None)),
        ("bin__has_agent", ("has_agent", None)),
        ("cat__country_PRT", ("country", "PRT")),
        ("cat__country_infrequent_sklearn", ("country", config.OTHER_OPTION)),
        ("cat__agent_9", ("agent", "9")),
    ],
)
def test_readable_feature(name: str, expected: tuple[str, str | None]) -> None:
    """Transformed names map back to input column and category."""
    assert features.readable_feature(name, GROUPS) == expected
