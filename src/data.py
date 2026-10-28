"""Load the raw dataset and clean it (invalid rows, duplicates, missing values, outliers).

Adapted from Task 2 (hotel-booking-preprocessing): ``load_data.py`` and
``cleaning.py`` merged into one module, logic unchanged. Every step is a small
pure function (DataFrame in, new DataFrame out); :func:`clean_data` runs them in
a fixed order and returns a per-step report. Reasoning: ``docs/decisions.md``.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, treating the mirror's ``"NULL"`` strings as missing.

    Args:
        path: Location of the raw CSV.

    Returns:
        The raw bookings as a DataFrame (unmodified apart from NA parsing).

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the target column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run `python -m src.download_data`.")

    df = pd.read_csv(path, na_values=config.MISSING_TOKENS, low_memory=False)
    if config.TARGET not in df.columns:
        raise ValueError(f"Target column '{config.TARGET}' not found in {path.name}.")

    logger.info("Loaded %s rows x %s columns from %s", f"{len(df):,}", df.shape[1], path.name)
    return df


def data_quality_snapshot(df: pd.DataFrame) -> dict[str, float]:
    """Count the known data-quality issues without changing the data.

    Args:
        df: Raw bookings DataFrame.

    Returns:
        Mapping of issue name to count (or value for ``adr_max``). The
        cancellation rate is included only when the target column is present
        (uploaded files may not have it).

    Raises:
        ValueError: If a required column is missing.
    """
    required = [*config.GUEST_COLUMNS, "adr"]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    total_guests = df[config.GUEST_COLUMNS].sum(axis=1, min_count=1)
    snapshot: dict[str, float] = {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
        "zero_guest_bookings": int((total_guests == 0).sum()),
        "negative_adr": int((df["adr"] < 0).sum()),
        "zero_adr": int((df["adr"] == 0).sum()),
        "adr_max": float(df["adr"].max()),
    }
    if config.TARGET in df.columns:
        snapshot["cancellation_rate_pct"] = round(float(df[config.TARGET].mean() * 100), 2)
    return snapshot


CleaningStep = Callable[[pd.DataFrame], pd.DataFrame]

REQUIRED_COLUMNS: list[str] = [
    *config.GUEST_COLUMNS,
    "adr",
    "agent",
    "company",
    "country",
    "meal",
    *config.MAX_PLAUSIBLE_COUNTS,
]


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise if any required column is missing.

    Args:
        df: DataFrame to check.
        required: Column names that must be present.

    Raises:
        ValueError: If one or more columns are absent.
    """
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Invalid rows
# ---------------------------------------------------------------------------
def drop_zero_guest_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bookings with no adults, children or babies (not a real stay).

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame without zero-guest rows.
    """
    total_guests = df[config.GUEST_COLUMNS].fillna(0).sum(axis=1)
    invalid = total_guests == 0
    logger.info("Zero-guest bookings dropped: %s", f"{invalid.sum():,}")
    return df[~invalid]


def drop_negative_adr(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a negative average daily rate (a price cannot be negative).

    Zero ADR is kept on purpose: it matches complimentary and day-use stays.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame without negative-ADR rows.
    """
    invalid = df["adr"] < 0
    logger.info("Negative-ADR rows dropped: %s", f"{invalid.sum():,}")
    return df[~invalid]


def drop_implausible_counts(
    df: pd.DataFrame, max_counts: dict[str, int] = config.MAX_PLAUSIBLE_COUNTS
) -> pd.DataFrame:
    """Drop rows where a count column exceeds its plausible maximum (entry errors).

    Rows are dropped rather than capped because the true value is unknown
    (e.g. ``adults = 55`` could have meant 5 or 2).

    Args:
        df: Bookings DataFrame.
        max_counts: Column name -> highest plausible value.

    Returns:
        DataFrame without implausible rows.
    """
    invalid = pd.Series(False, index=df.index)
    for column, max_value in max_counts.items():
        too_high = df[column] > max_value
        logger.info("  %-28s > %d: %s rows", column, max_value, f"{too_high.sum():,}")
        invalid |= too_high
    logger.info("Implausible-count rows dropped: %s", f"{invalid.sum():,}")
    return df[~invalid]


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence.

    Must run before any column is dropped or recoded, otherwise rows that only
    differed in that column would wrongly become duplicates.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with unique rows.
    """
    duplicated = df.duplicated(keep="first")
    logger.info("Exact duplicate rows dropped: %s", f"{duplicated.sum():,}")
    return df[~duplicated]


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
def replace_company_with_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the ~94%-empty ``company`` ID with a binary ``has_company`` flag.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with ``has_company`` added and ``company`` removed.
    """
    has_company = df["company"].notna().astype(int)
    logger.info("has_company = 1 for %s bookings; company column dropped", f"{has_company.sum():,}")
    return df.assign(has_company=has_company).drop(columns="company")


def fill_agent(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``has_agent`` flag and fill missing agent IDs with the 'no agent' code.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with integer ``agent`` and a ``has_agent`` flag.
    """
    has_agent = df["agent"].notna().astype(int)
    logger.info("agent missing -> %d: %s rows", config.NO_AGENT_ID, f"{(has_agent == 0).sum():,}")
    return df.assign(
        agent=df["agent"].fillna(config.NO_AGENT_ID).astype(int),
        has_agent=has_agent,
    )


def fill_country(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing country codes with an explicit 'Unknown' category.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with no missing ``country``.
    """
    logger.info(
        "country missing -> '%s': %s rows",
        config.UNKNOWN_COUNTRY,
        f"{df['country'].isna().sum():,}",
    )
    return df.assign(country=df["country"].fillna(config.UNKNOWN_COUNTRY))


def fill_children(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``children`` with 0 and restore its integer type.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with integer ``children``.
    """
    logger.info("children missing -> 0: %s rows", f"{df['children'].isna().sum():,}")
    return df.assign(children=df["children"].fillna(0).astype(int))


# ---------------------------------------------------------------------------
# Category consistency
# ---------------------------------------------------------------------------
def standardize_meal(df: pd.DataFrame) -> pd.DataFrame:
    """Merge meal labels that mean the same thing ("Undefined" -> "SC").

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with consistent ``meal`` labels.
    """
    affected = df["meal"].isin(config.MEAL_REPLACEMENTS).sum()
    logger.info("meal relabelled %s: %s rows", config.MEAL_REPLACEMENTS, f"{affected:,}")
    return df.assign(meal=df["meal"].replace(config.MEAL_REPLACEMENTS))


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
def compute_adr_cap(df: pd.DataFrame, quantile: float = config.ADR_CAP_QUANTILE) -> float:
    """Return the ADR value at the given upper quantile.

    Args:
        df: Bookings DataFrame.
        quantile: Upper quantile between 0 and 1.

    Returns:
        The cap value.

    Raises:
        ValueError: If ``quantile`` is outside (0, 1].
    """
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    return float(df["adr"].quantile(quantile))


def cap_adr(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Cap (winsorise) ADR values above ``cap``.

    Args:
        df: Bookings DataFrame.
        cap: Highest allowed ADR; larger values are set to this value.

    Returns:
        DataFrame with capped ``adr``.
    """
    above = df["adr"] > cap
    logger.info(
        "ADR capped at %.2f: %s rows, old max %.2f", cap, f"{above.sum():,}", df["adr"].max()
    )
    return df.assign(adr=df["adr"].clip(upper=cap))


# ---------------------------------------------------------------------------
# Checks and orchestration
# ---------------------------------------------------------------------------
def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns that reveal the outcome (``reservation_status`` and its date).

    Changed from Task 2, which kept them for EDA: this project only models, so they
    are removed as soon as cleaning ends and can never reach a model.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame without the leakage columns.
    """
    present = [col for col in config.LEAKAGE_COLUMNS if col in df.columns]
    if present:
        logger.info("Leakage columns dropped: %s", present)
    return df.drop(columns=present)


def check_no_missing(df: pd.DataFrame) -> None:
    """Raise if any missing values remain after cleaning.

    Args:
        df: Cleaned DataFrame.

    Raises:
        ValueError: If any column still contains missing values.
    """
    remaining = df.isna().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        raise ValueError(f"Missing values remain after cleaning: {remaining.to_dict()}")


def _count_changed_values(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """Count cells that changed value between two frames on shared rows/columns.

    Args:
        before: Frame before a step.
        after: Frame after the step.

    Returns:
        Number of changed cells (NaN -> value counts as a change).
    """
    columns = before.columns.intersection(after.columns)
    old = before.loc[after.index, columns]
    new = after[columns]
    unchanged = old.eq(new) | (old.isna() & new.isna())
    return int((~unchanged).to_numpy().sum())


CLEANING_STEPS: list[tuple[str, CleaningStep]] = [
    ("Drop zero-guest bookings", drop_zero_guest_bookings),
    ("Drop negative ADR", drop_negative_adr),
    ("Drop implausible counts", drop_implausible_counts),
    ("Drop exact duplicates", drop_duplicate_rows),
    ("company -> has_company flag", replace_company_with_flag),
    ("Fill agent + has_agent flag", fill_agent),
    ("Fill country 'Unknown'", fill_country),
    ("Fill children 0", fill_children),
    ("meal Undefined -> SC", standardize_meal),
]
CAP_STEP_NAME: str = "Cap extreme ADR"


class CleaningResult(NamedTuple):
    """Output of :func:`clean_data`."""

    data: pd.DataFrame
    report: pd.DataFrame
    adr_cap: float


def _report_row(name: str, before: pd.DataFrame, after: pd.DataFrame) -> dict[str, object]:
    """Describe what one cleaning step changed.

    Args:
        name: Step name.
        before: Frame before the step.
        after: Frame after the step.

    Returns:
        One row of the step report.
    """
    added = sorted(set(after.columns) - set(before.columns))
    removed = sorted(set(before.columns) - set(after.columns))
    return {
        "step": name,
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_removed": len(before) - len(after),
        "values_changed": _count_changed_values(before, after),
        "columns": ", ".join([f"+{c}" for c in added] + [f"-{c}" for c in removed]),
    }


def clean_data(df: pd.DataFrame, adr_cap: float | None = None) -> CleaningResult:
    """Run every cleaning step in order and record what each one changed.

    Args:
        df: Raw bookings DataFrame (as returned by ``load_raw_data``).
        adr_cap: Fixed ADR cap to apply. ``None`` computes it from this data
            (``config.ADR_CAP_QUANTILE``); pass the reference cap when cleaning
            *new* data so it is treated exactly like the original dataset.

    Returns:
        ``CleaningResult(data, report, adr_cap)``. The report has one row per
        step with rows before/after, rows removed, values changed and column changes.

    Raises:
        ValueError: If required columns are missing or nulls remain at the end.
    """
    validate_columns(df, REQUIRED_COLUMNS)
    report_rows = []
    current = df
    for name, step in CLEANING_STEPS:
        logger.info("-> %s", name)
        result = step(current)
        report_rows.append(_report_row(name, current, result))
        current = result

    logger.info("-> %s", CAP_STEP_NAME)
    cap = compute_adr_cap(current) if adr_cap is None else adr_cap
    result = cap_adr(current, cap)
    report_rows.append(_report_row(CAP_STEP_NAME, current, result))
    current = result

    check_no_missing(current)
    cleaned = drop_leakage_columns(current).reset_index(drop=True)
    logger.info(
        "Cleaning done: %s -> %s rows (%s removed, %.1f%%)",
        f"{len(df):,}",
        f"{len(cleaned):,}",
        f"{len(df) - len(cleaned):,}",
        (len(df) - len(cleaned)) / max(len(df), 1) * 100,
    )
    return CleaningResult(cleaned, pd.DataFrame(report_rows).set_index("step"), cap)


def save_clean_data(df: pd.DataFrame, path: Path = config.CLEAN_DATA_PATH) -> Path:
    """Write the cleaned data to CSV.

    Args:
        df: Cleaned bookings DataFrame.
        path: Output CSV path.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        "Saved cleaned data: %s (%s rows x %s columns)", path.name, f"{len(df):,}", df.shape[1]
    )
    return path
