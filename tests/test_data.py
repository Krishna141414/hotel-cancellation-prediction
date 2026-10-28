"""Tests for src.data (loading and cleaning; adapted from Task 2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src import data as cleaning


@pytest.fixture
def bookings() -> pd.DataFrame:
    """Six small bookings covering every issue the cleaning steps handle.

    Row 0: valid. Row 1: exact duplicate of row 0. Row 2: zero guests.
    Row 3: negative ADR. Row 4: 55 adults (entry error). Row 5: valid with
    missing agent/company/country/children, meal "Undefined" and huge ADR.
    """
    return pd.DataFrame(
        {
            "is_canceled": [0, 0, 1, 0, 1, 1],
            "adults": [2, 2, 0, 2, 55, 1],
            "children": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            "babies": [0, 0, 0, 0, 0, 0],
            "required_car_parking_spaces": [0, 0, 0, 0, 0, 1],
            "adr": [100.0, 100.0, 80.0, -6.4, 0.0, 5400.0],
            "agent": [9.0, 9.0, 9.0, 9.0, np.nan, np.nan],
            "company": [np.nan, np.nan, np.nan, 40.0, np.nan, 40.0],
            "country": ["PRT", "PRT", "GBR", "PRT", "ESP", np.nan],
            "meal": ["BB", "BB", "BB", "HB", "BB", "Undefined"],
            "reservation_status": ["Check-Out"] * 2
            + ["Canceled"]
            + ["Check-Out"]
            + ["Canceled"] * 2,
        }
    )


def test_drop_zero_guest_bookings(bookings: pd.DataFrame) -> None:
    """Only the row with 0 adults, 0 children and 0 babies is removed."""
    result = cleaning.drop_zero_guest_bookings(bookings)
    assert len(result) == 5
    assert 2 not in result.index


def test_drop_zero_guest_treats_missing_children_as_zero() -> None:
    """A row with missing children but real adults is kept."""
    df = pd.DataFrame({"adults": [1], "children": [np.nan], "babies": [0]})
    assert len(cleaning.drop_zero_guest_bookings(df)) == 1


def test_drop_negative_adr_keeps_zero(bookings: pd.DataFrame) -> None:
    """Negative ADR is dropped but zero ADR (complimentary stay) is kept."""
    result = cleaning.drop_negative_adr(bookings)
    assert (result["adr"] >= 0).all()
    assert (result["adr"] == 0).any()


def test_drop_implausible_counts(bookings: pd.DataFrame) -> None:
    """Rows above the per-column plausible maximum are removed."""
    result = cleaning.drop_implausible_counts(bookings, {"adults": 4})
    assert 4 not in result.index
    assert len(result) == 5


def test_drop_duplicate_rows_keeps_first(bookings: pd.DataFrame) -> None:
    """Exact duplicates are removed and the first occurrence survives."""
    result = cleaning.drop_duplicate_rows(bookings)
    assert len(result) == 5
    assert 0 in result.index and 1 not in result.index


def test_replace_company_with_flag(bookings: pd.DataFrame) -> None:
    """The company ID is replaced by a 0/1 flag."""
    result = cleaning.replace_company_with_flag(bookings)
    assert "company" not in result.columns
    assert result["has_company"].tolist() == [0, 0, 0, 1, 0, 1]


def test_fill_agent(bookings: pd.DataFrame) -> None:
    """Missing agents become the no-agent code and are flagged."""
    result = cleaning.fill_agent(bookings)
    assert result["agent"].tolist() == [9, 9, 9, 9, config.NO_AGENT_ID, config.NO_AGENT_ID]
    assert result["has_agent"].tolist() == [1, 1, 1, 1, 0, 0]
    assert result["agent"].dtype.kind == "i"


def test_fill_country_and_children(bookings: pd.DataFrame) -> None:
    """Missing country becomes 'Unknown' and children gets 0 as an integer column."""
    result = cleaning.fill_children(cleaning.fill_country(bookings))
    assert result.loc[5, "country"] == config.UNKNOWN_COUNTRY
    assert result.loc[5, "children"] == 0
    assert result["children"].dtype.kind == "i"


def test_standardize_meal(bookings: pd.DataFrame) -> None:
    """'Undefined' meal is merged into 'SC'."""
    result = cleaning.standardize_meal(bookings)
    assert "Undefined" not in result["meal"].tolist()
    assert result.loc[5, "meal"] == "SC"


def test_cap_adr_limits_extreme_values() -> None:
    """Values above the quantile cap are capped; values below are untouched."""
    df = pd.DataFrame({"adr": [float(v) for v in range(1, 101)] + [5400.0]})
    cap = cleaning.compute_adr_cap(df, quantile=0.99)
    result = cleaning.cap_adr(df, cap)
    assert result["adr"].max() == cap < 5400.0
    assert result["adr"].iloc[0] == 1.0


def test_compute_adr_cap_rejects_bad_quantile() -> None:
    """Quantiles outside (0, 1] raise a ValueError."""
    with pytest.raises(ValueError, match="quantile"):
        cleaning.compute_adr_cap(pd.DataFrame({"adr": [1.0]}), quantile=1.5)


def test_check_no_missing_raises() -> None:
    """Remaining nulls are reported with the column name."""
    with pytest.raises(ValueError, match="agent"):
        cleaning.check_no_missing(pd.DataFrame({"agent": [1.0, np.nan]}))


def test_validate_columns_raises() -> None:
    """Missing required columns are reported."""
    with pytest.raises(ValueError, match="Missing required columns"):
        cleaning.validate_columns(pd.DataFrame({"adr": [1.0]}), ["adr", "agent"])


def test_clean_data_end_to_end(bookings: pd.DataFrame) -> None:
    """The full run removes invalid rows and duplicates and leaves no nulls."""
    cleaned, report, _ = cleaning.clean_data(bookings)
    # Rows 1 (duplicate), 2 (zero guests), 3 (negative ADR), 4 (55 adults) go.
    assert len(cleaned) == 2
    assert cleaned.isna().sum().sum() == 0
    assert {"has_company", "has_agent"} <= set(cleaned.columns)
    assert "company" not in cleaned.columns
    assert "reservation_status" not in cleaned.columns  # leakage dropped
    assert report["rows_removed"].sum() == 4
    expected_steps = [name for name, _ in cleaning.CLEANING_STEPS] + [cleaning.CAP_STEP_NAME]
    assert list(report.index) == expected_steps


def test_clean_data_reuses_a_fixed_adr_cap(bookings: pd.DataFrame) -> None:
    """New data is capped with the reference cap, not one computed from itself."""
    result = cleaning.clean_data(bookings, adr_cap=150.0)
    assert result.adr_cap == 150.0
    assert result.data["adr"].max() == 150.0


def test_clean_data_without_target_column(bookings: pd.DataFrame) -> None:
    """Data without the target (e.g. new bookings) can still be cleaned."""
    cleaned = cleaning.clean_data(bookings.drop(columns="is_canceled")).data
    assert config.TARGET not in cleaned.columns
    assert cleaned.isna().sum().sum() == 0


def test_drop_leakage_columns() -> None:
    """Both outcome columns are removed; other columns stay."""
    df = pd.DataFrame(
        {"reservation_status": ["Canceled"], "reservation_status_date": ["x"], "a": [1]}
    )
    assert list(cleaning.drop_leakage_columns(df).columns) == ["a"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """Write a tiny CSV that mimics the mirror's quirks ("NULL" and "NA" strings)."""
    content = (
        "is_canceled,adults,children,babies,adr,agent,country\n"
        "0,2,0,0,100.0,9,PRT\n"
        "1,2,NA,0,-6.4,NULL,NULL\n"
        "0,0,0,0,0.0,NULL,GBR\n"
        "0,2,0,0,100.0,9,PRT\n"
    )
    path = tmp_path / "bookings.csv"
    path.write_text(content)
    return path


def test_load_raw_data_treats_null_string_as_missing(raw_csv: Path) -> None:
    """The literal string "NULL" must become NaN, not a category called "NULL"."""
    df = cleaning.load_raw_data(raw_csv)
    assert df["agent"].isna().sum() == 2
    assert df["country"].isna().sum() == 1
    assert df["children"].isna().sum() == 1
    assert "NULL" not in df["country"].astype(str).tolist()


def test_load_raw_data_missing_file_raises(tmp_path: Path) -> None:
    """A clear error is raised when the CSV is absent."""
    with pytest.raises(FileNotFoundError):
        cleaning.load_raw_data(tmp_path / "nope.csv")


def test_data_quality_snapshot_counts_known_issues(raw_csv: Path) -> None:
    """Duplicates, zero-guest bookings and negative ADR are counted correctly."""
    snapshot = cleaning.data_quality_snapshot(cleaning.load_raw_data(raw_csv))
    assert snapshot["duplicate_rows"] == 1
    assert snapshot["zero_guest_bookings"] == 1
    assert snapshot["negative_adr"] == 1
    assert snapshot["cancellation_rate_pct"] == 25.0


def test_data_quality_snapshot_requires_columns() -> None:
    """Missing required columns raise a clear ValueError."""
    with pytest.raises(ValueError, match="Missing required columns"):
        cleaning.data_quality_snapshot(pd.DataFrame({"adr": [1.0]}))
