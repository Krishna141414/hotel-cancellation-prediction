"""Feature engineering, feature selection checks and preprocessing transformers.

Adapted from Task 2 (hotel-booking-preprocessing): the engineering and selection
functions of ``features.py`` are unchanged; the split and ``ColumnTransformer``
helpers come from Task 2's ``pipeline.py``. New here: a tree-friendly
preprocessor (no scaling), a time-ordered split and readable feature names.

Every engineered feature is computed *within a single row* (no dataset-wide
statistics), so creating them before the train/test split cannot leak
information. All fitted statistics (scaling, one-hot categories, log choice)
live inside the preprocessors and are fitted on training folds only.
"""

import calendar
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)

MONTH_NAME_TO_NUMBER: dict[str, int] = {
    name: number for number, name in enumerate(calendar.month_name) if name
}

FEATURE_INPUT_COLUMNS: list[str] = [
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    *config.GUEST_COLUMNS,
    "adr",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "country",
    "previous_cancellations",
    "previous_bookings_not_canceled",
]


def _require(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise if any of ``columns`` is missing from ``df``.

    Args:
        df: DataFrame to check.
        columns: Required column names.

    Raises:
        ValueError: If a column is missing.
    """
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {missing}")


def add_stay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``total_nights``, ``total_guests`` and ``has_children``.

    Args:
        df: Cleaned bookings DataFrame.

    Returns:
        DataFrame with the three new columns.
    """
    _require(df, ["stays_in_weekend_nights", "stays_in_week_nights", *config.GUEST_COLUMNS])
    return df.assign(
        total_nights=df["stays_in_weekend_nights"] + df["stays_in_week_nights"],
        total_guests=df[config.GUEST_COLUMNS].sum(axis=1),
        has_children=((df["children"] + df["babies"]) > 0).astype(int),
    )


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``adr_per_person`` (per paying guest) and ``stay_value`` (whole stay).

    Babies usually stay free, so the per-person price divides by adults + children.
    Requires ``total_nights`` (see :func:`add_stay_features`).

    Args:
        df: Bookings DataFrame with ``total_nights``.

    Returns:
        DataFrame with the two price columns.

    Raises:
        ValueError: If a booking has no paying guests (would divide by zero).
    """
    _require(df, ["adr", "adults", "children", "total_nights"])
    paying_guests = df["adults"] + df["children"]
    if (paying_guests <= 0).any():
        raise ValueError("Found bookings with 0 paying guests; run cleaning first.")
    return df.assign(
        adr_per_person=(df["adr"] / paying_guests).round(2),
        stay_value=(df["adr"] * df["total_nights"]).round(2),
    )


def add_arrival_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month number, weekday (0 = Monday) and season of the arrival date.

    Args:
        df: Bookings DataFrame with the split arrival-date columns.

    Returns:
        DataFrame with ``arrival_month_num``, ``arrival_weekday``, ``arrival_season``.

    Raises:
        ValueError: If a month name is not recognised or a date is invalid.
    """
    _require(df, ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"])
    month_num = df["arrival_date_month"].map(MONTH_NAME_TO_NUMBER)
    if month_num.isna().any():
        bad = sorted(df.loc[month_num.isna(), "arrival_date_month"].astype(str).unique())
        raise ValueError(f"Unrecognised arrival_date_month values: {bad}")

    arrival_date = pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["arrival_date_year"],
                "month": month_num,
                "day": df["arrival_date_day_of_month"],
            }
        ),
        errors="raise",
    )
    return df.assign(
        arrival_month_num=month_num.astype(int),
        arrival_weekday=arrival_date.dt.day_of_week,
        arrival_season=month_num.map(config.SEASON_BY_MONTH),
    )


def add_guest_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_domestic`` and ``previous_cancel_ratio``.

    ``previous_cancel_ratio`` is the share of the guest's earlier bookings that
    were canceled (0 for guests with no history).

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with the two new columns.
    """
    _require(df, ["country", "previous_cancellations", "previous_bookings_not_canceled"])
    previous_total = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    ratio = (df["previous_cancellations"] / previous_total.where(previous_total > 0)).fillna(0)
    return df.assign(
        is_domestic=(df["country"] == config.DOMESTIC_COUNTRY).astype(int),
        previous_cancel_ratio=ratio.round(3),
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every feature-engineering step on cleaned data.

    Args:
        df: Cleaned bookings DataFrame.

    Returns:
        DataFrame with all columns in ``config.ENGINEERED_FEATURES`` added.
    """
    _require(df, FEATURE_INPUT_COLUMNS)
    result = add_guest_history_features(
        add_arrival_features(add_price_features(add_stay_features(df)))
    )
    logger.info(
        "Engineered %d features: %s", len(config.ENGINEERED_FEATURES), config.ENGINEERED_FEATURES
    )
    logger.info(
        "Shape after feature engineering: %s rows x %s columns", f"{len(result):,}", result.shape[1]
    )
    return result


# ---------------------------------------------------------------------------
# Feature selection checks (run on TRAINING rows only)
# ---------------------------------------------------------------------------
def target_purity(df: pd.DataFrame, column: str, target: str = config.TARGET) -> float:
    """Share of rows a column predicts correctly by majority vote per value.

    1.0 means every value of ``column`` maps to a single target class - the
    column alone "knows" the answer, a typical sign of leakage. A useless column
    scores the majority-class share (0.725 here).

    Args:
        df: Training rows with ``column`` and ``target``.
        column: Column to check.
        target: Binary target column.

    Returns:
        Weighted majority-class share across the column's values.
    """
    _require(df, [column, target])
    rate = df.groupby(column, observed=True)[target].agg(["mean", "size"])
    majority = rate["mean"].where(rate["mean"] >= 0.5, 1 - rate["mean"])
    return float((majority * rate["size"]).sum() / rate["size"].sum())


def find_leakage_columns(
    df: pd.DataFrame, columns: list[str], threshold: float = config.LEAKAGE_PURITY_THRESHOLD
) -> dict[str, float]:
    """Flag columns whose values alone (almost) perfectly predict the target.

    Args:
        df: Training rows.
        columns: Categorical-style columns to check.
        threshold: Purity at or above which a column is flagged.

    Returns:
        Flagged column -> purity.
    """
    purity = {column: round(target_purity(df, column), 4) for column in columns}
    return {column: value for column, value in purity.items() if value >= threshold}


def find_correlated_pairs(
    df: pd.DataFrame, columns: list[str], threshold: float = config.CORRELATION_THRESHOLD
) -> pd.DataFrame:
    """List numeric column pairs with absolute Pearson correlation above ``threshold``.

    Args:
        df: Training rows.
        columns: Numeric columns to compare.
        threshold: Minimum absolute correlation to report.

    Returns:
        DataFrame with ``feature_a``, ``feature_b`` and ``abs_corr``, strongest first.
    """
    _require(df, columns)
    corr = df[columns].corr().abs()
    pairs = [
        {"feature_a": a, "feature_b": b, "abs_corr": round(float(corr.loc[a, b]), 3)}
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
        if corr.loc[a, b] > threshold
    ]
    table = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_corr"])
    return table.sort_values("abs_corr", ascending=False, ignore_index=True)


def find_near_constant(
    df: pd.DataFrame, columns: list[str], threshold: float = config.NEAR_CONSTANT_THRESHOLD
) -> dict[str, float]:
    """Flag columns where a single value covers at least ``threshold`` of the rows.

    Args:
        df: Training rows.
        columns: Columns to check.
        threshold: Share of the most common value at which a column is flagged.

    Returns:
        Flagged column -> share of its most common value.
    """
    _require(df, columns)
    shares = {c: round(float(df[c].value_counts(normalize=True).iloc[0]), 4) for c in columns}
    return {column: share for column, share in shares.items() if share >= threshold}


def log_selection_report(
    train_rows: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> dict[str, object]:
    """Run every selection check on the training rows and log the findings.

    Args:
        train_rows: Training split (all columns, incl. the leakage columns).
        numeric: Numeric + binary candidate columns.
        categorical: Categorical candidates plus known leakage columns to check.

    Returns:
        ``{"leakage": ..., "correlated": ..., "near_constant": ...}``.
    """
    report: dict[str, object] = {
        "leakage": find_leakage_columns(train_rows, categorical),
        "correlated": find_correlated_pairs(train_rows, numeric),
        "near_constant": find_near_constant(train_rows, numeric),
    }
    logger.info(
        "Leakage check (purity >= %.2f): %s", config.LEAKAGE_PURITY_THRESHOLD, report["leakage"]
    )
    logger.info(
        "Redundant pairs (|r| > %.2f):\n%s",
        config.CORRELATION_THRESHOLD,
        report["correlated"].to_string(index=False),
    )
    logger.info(
        "Near-constant (>= %.0f%% one value): %s",
        config.NEAR_CONSTANT_THRESHOLD * 100,
        report["near_constant"],
    )
    return report


def validate_selection(
    train_rows: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> None:
    """Fail loudly if the final feature set still contains leakage or redundant pairs.

    Args:
        train_rows: Training split.
        numeric: Final numeric + binary features.
        categorical: Final categorical features.

    Raises:
        ValueError: If a final feature is flagged as leakage or part of a redundant pair,
            or a known leakage column is still selected.
    """
    selected = set(numeric) | set(categorical)
    known = selected & set(config.LEAKAGE_COLUMNS) | selected & set(config.DROPPED_FEATURES)
    leakage = find_leakage_columns(train_rows, categorical)
    correlated = find_correlated_pairs(train_rows, numeric)
    if known or leakage or not correlated.empty:
        raise ValueError(
            f"Feature selection check failed: dropped/leaky={sorted(known)}, "
            f"purity={leakage}, redundant pairs={correlated.to_dict('records')}"
        )
    logger.info("Selection validated: no leakage, no redundant pairs in %d features", len(selected))


# ---------------------------------------------------------------------------
# Model inputs and splits (from Task 2's pipeline.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureGroups:
    """Model input columns, grouped by how they are preprocessed."""

    numeric: list[str]
    binary: list[str]
    categorical: list[str]

    @property
    def all(self) -> list[str]:
        """Every input column, in transformer order."""
        return [*self.numeric, *self.binary, *self.categorical]

    @classmethod
    def from_config(cls) -> "FeatureGroups":
        """Build the *candidate* groups from the lists in ``src/config.py``."""
        return cls(
            numeric=list(config.NUMERIC_FEATURES),
            binary=list(config.BINARY_FEATURES),
            categorical=list(config.CATEGORICAL_FEATURES),
        )

    @classmethod
    def selected(cls) -> "FeatureGroups":
        """The final model inputs: candidates minus ``config.DROPPED_FEATURES``."""
        return cls.from_config().without(config.DROPPED_FEATURES)

    def without(self, columns: list[str] | dict[str, str]) -> "FeatureGroups":
        """Return a copy with ``columns`` removed from every group.

        Args:
            columns: Column names (or a name -> reason mapping) to drop.

        Returns:
            The reduced feature groups.
        """
        drop = set(columns)
        return FeatureGroups(
            numeric=[c for c in self.numeric if c not in drop],
            binary=[c for c in self.binary if c not in drop],
            categorical=[c for c in self.categorical if c not in drop],
        )

    def with_columns(self, columns: list[str]) -> "FeatureGroups":
        """Return a copy with candidate ``columns`` added back to their group.

        Args:
            columns: Candidate column names (from ``src/config.py``) to add.

        Returns:
            The extended feature groups.

        Raises:
            ValueError: If a column is not a known candidate.
        """
        candidates = FeatureGroups.from_config()
        unknown = sorted(set(columns) - set(candidates.all))
        if unknown:
            raise ValueError(f"Not candidate features: {unknown}")
        extra = [c for c in columns if c not in self.all]
        return FeatureGroups(
            numeric=self.numeric + [c for c in extra if c in candidates.numeric],
            binary=self.binary + [c for c in extra if c in candidates.binary],
            categorical=self.categorical + [c for c in extra if c in candidates.categorical],
        )


def arrival_dates(df: pd.DataFrame) -> pd.Series:
    """Rebuild the arrival date from the split year / month-name / day columns.

    Args:
        df: Bookings with ``arrival_date_year``, ``arrival_date_month`` and
            ``arrival_date_day_of_month``.

    Returns:
        Arrival dates as a datetime Series aligned with ``df``.
    """
    _require(df, ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"])
    return pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["arrival_date_year"],
                "month": df["arrival_date_month"].map(MONTH_NAME_TO_NUMBER),
                "day": df["arrival_date_day_of_month"],
            }
        )
    )


def split_frame(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole rows into stratified train/test frames (same split as Task 2).

    Args:
        df: Bookings with the target column.
        test_size: Share of rows held out for testing.
        random_state: Seed for a reproducible split.

    Returns:
        ``(train_rows, test_rows)``.
    """
    _require(df, [config.TARGET])
    train_rows, test_rows = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[config.TARGET]
    )
    return train_rows, test_rows


def temporal_split(
    df: pd.DataFrame, test_share: float = config.TEMPORAL_TEST_SHARE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by arrival date: earlier arrivals train, the latest ``test_share`` test.

    All rows with the same arrival date land on the same side, so no day is split.

    Args:
        df: Bookings with the arrival-date columns and the target.
        test_share: Approximate share of (latest) rows used for testing.

    Returns:
        ``(train_rows, test_rows, cutoff)`` where test rows arrive on/after ``cutoff``.

    Raises:
        ValueError: If ``test_share`` is not between 0 and 1.
    """
    if not 0 < test_share < 1:
        raise ValueError(f"test_share must be in (0, 1), got {test_share}")
    dates = arrival_dates(df)
    cutoff = dates.quantile(1 - test_share).normalize()
    is_test = dates >= cutoff
    return df[~is_test], df[is_test], cutoff


# ---------------------------------------------------------------------------
# Preprocessing transformers (fitted inside each model pipeline)
# ---------------------------------------------------------------------------
def choose_log_columns(
    x_train: pd.DataFrame,
    numeric: list[str],
    skew_threshold: float = config.LOG_SKEW_THRESHOLD,
    max_ratio: float = config.LOG_SKEW_MAX_RATIO,
) -> list[str]:
    """Pick numeric columns that ``log1p`` clearly de-skews (training data only).

    A column qualifies when its skew is above ``skew_threshold`` and ``log1p``
    shrinks the absolute skew to at most ``max_ratio`` of the original. Columns
    with negative values are never log-transformed.

    Args:
        x_train: Training features.
        numeric: Candidate numeric columns.
        skew_threshold: Minimum skew to consider a transform.
        max_ratio: Required improvement (after / before absolute skew).

    Returns:
        Columns to log-transform, in the original order.
    """
    chosen = []
    for column in numeric:
        values = x_train[column]
        if (values < 0).any():
            continue
        before, after = values.skew(), np.log1p(values).skew()
        if before > skew_threshold and abs(after) <= max_ratio * abs(before):
            chosen.append(column)
    logger.debug("log1p chosen for %d columns: %s", len(chosen), chosen)
    return chosen


class SkewLog1p(TransformerMixin, BaseEstimator):
    """Apply ``log1p`` to the numeric columns that the Task 2 skew rule selects.

    Task 2 chose the columns once on the whole training set. Here the choice is
    made inside ``fit``, so during cross-validation each training fold decides
    for itself and the validation fold never influences it.

    Args:
        skew_threshold: Minimum skew to consider a transform.
        max_ratio: Required improvement (after / before absolute skew).
    """

    def __init__(
        self,
        skew_threshold: float = config.LOG_SKEW_THRESHOLD,
        max_ratio: float = config.LOG_SKEW_MAX_RATIO,
    ) -> None:
        """Store the rule parameters (sklearn convention: no logic here)."""
        self.skew_threshold = skew_threshold
        self.max_ratio = max_ratio

    def fit(self, x: pd.DataFrame, y: object = None) -> "SkewLog1p":
        """Choose the columns to transform from the training data.

        Args:
            x: Numeric training columns.
            y: Ignored.

        Returns:
            The fitted transformer.
        """
        frame = pd.DataFrame(x)
        self.columns_ = list(frame.columns)
        self.log_columns_ = choose_log_columns(
            frame, self.columns_, self.skew_threshold, self.max_ratio
        )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply ``log1p`` to the chosen columns.

        Args:
            x: Numeric columns (same as in ``fit``).

        Returns:
            Transformed copy.
        """
        frame = pd.DataFrame(x, columns=self.columns_).copy()
        for column in self.log_columns_:
            frame[column] = np.log1p(frame[column].clip(lower=0))
        return frame

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Column names are unchanged (one-to-one transform).

        Args:
            input_features: Ignored.

        Returns:
            The input column names.
        """
        return np.asarray(self.columns_, dtype=object)


def to_string(values: pd.DataFrame) -> pd.DataFrame:
    """Cast categories to text so numeric IDs (e.g. agent 9) are treated as labels.

    A named module-level function (not a lambda) so the fitted pipeline can be
    saved with joblib.

    Args:
        values: Categorical input columns.

    Returns:
        The same columns as strings.
    """
    return pd.DataFrame(values).astype(str)


def _categorical_pipeline() -> Pipeline:
    """Most-frequent impute -> text -> one-hot, rare categories grouped as "infrequent".

    Returns:
        The (unfitted) categorical pipeline.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("to_string", FunctionTransformer(to_string, feature_names_out="one-to-one")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=config.MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_scaled_preprocessor(groups: FeatureGroups) -> ColumnTransformer:
    """Preprocessing for distance/linear models (Logistic Regression, KNN).

    Task 2's transformer: numeric -> median impute -> log1p (skewed columns only)
    -> standard scale; binary flags -> impute only; categorical -> one-hot.

    Args:
        groups: Model input columns by type.

    Returns:
        An unfitted ColumnTransformer that outputs a pandas DataFrame.
    """
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", SkewLog1p()),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, groups.numeric),
            ("bin", SimpleImputer(strategy="most_frequent"), groups.binary),
            ("cat", _categorical_pipeline(), groups.categorical),
        ],
        remainder="drop",
    )
    return preprocessor.set_output(transform="pandas")


def build_tree_preprocessor(groups: FeatureGroups) -> ColumnTransformer:
    """Preprocessing for tree models: same one-hot encoding, but no log or scaling.

    Trees split on thresholds, so monotonic transforms change nothing for them;
    keeping raw units makes every split readable ("lead_time <= 12 days").

    Args:
        groups: Model input columns by type.

    Returns:
        An unfitted ColumnTransformer that outputs a pandas DataFrame.
    """
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), groups.numeric),
            ("bin", SimpleImputer(strategy="most_frequent"), groups.binary),
            ("cat", _categorical_pipeline(), groups.categorical),
        ],
        remainder="drop",
    )
    return preprocessor.set_output(transform="pandas")


def readable_feature(name: str, groups: FeatureGroups) -> tuple[str, str | None]:
    """Map a transformed column name back to its input column (and category).

    Args:
        name: Output name such as ``"num__lead_time"`` or ``"cat__deposit_type_Non Refund"``.
        groups: The input columns used by the preprocessor.

    Returns:
        ``(input_column, category)``; ``category`` is ``None`` for numeric/binary
        columns and ``"Other"`` for the grouped infrequent categories.
    """
    bare = name.split("__", 1)[-1]
    if bare in groups.numeric or bare in groups.binary:
        return bare, None
    # Longest match first so a column name is never matched by a shorter prefix.
    for column in sorted(groups.categorical, key=len, reverse=True):
        if bare.startswith(f"{column}_"):
            category = bare[len(column) + 1 :]
            return column, config.OTHER_OPTION if category == "infrequent_sklearn" else category
    return bare, None
