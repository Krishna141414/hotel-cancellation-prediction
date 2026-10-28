"""Central configuration: paths, dataset source, columns, model settings and constants.

Every other module imports from here so there are no hardcoded paths or magic
numbers scattered around the codebase. The data/cleaning/feature sections are
carried over unchanged from Task 2 (hotel-booking-preprocessing).
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RAW_DATA_PATH: Path = RAW_DATA_DIR / "hotel_bookings.csv"
CLEAN_DATA_PATH: Path = PROCESSED_DATA_DIR / "hotel_bookings_clean.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
FINAL_MODEL_PATH: Path = MODELS_DIR / "final_model.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODEL_COMPARISON_PATH: Path = REPORTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Dataset source (public TidyTuesday mirror of Antonio, Almeida & Nunes, 2019)
# ---------------------------------------------------------------------------
_MIRROR_PATH: str = "data/2020/2020-02-11/hotels.csv"
DATA_URLS: tuple[str, ...] = (
    f"https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/{_MIRROR_PATH}",
    f"https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/{_MIRROR_PATH}",
)
DOWNLOAD_TIMEOUT_SECONDS: int = 60

EXPECTED_ROWS: int = 119_390
EXPECTED_COLUMNS: int = 32

# The mirror writes missing values in some columns as the literal string "NULL".
MISSING_TOKENS: list[str] = ["NULL"]

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
TARGET: str = "is_canceled"
CLASS_NAMES: tuple[str, str] = ("Not canceled", "Canceled")

# Known only after the stay/cancellation happens -> would leak the target.
LEAKAGE_COLUMNS: list[str] = ["reservation_status", "reservation_status_date"]

GUEST_COLUMNS: list[str] = ["adults", "children", "babies"]

# ---------------------------------------------------------------------------
# Cleaning rules (Task 2; reasoning in docs/decisions.md, section 2)
# ---------------------------------------------------------------------------
# Agent IDs start at 1, so 0 is a safe "booked without an agent" code.
NO_AGENT_ID: int = 0
UNKNOWN_COUNTRY: str = "Unknown"
# Per the dataset paper, "Undefined" and "SC" both mean no meal package.
MEAL_REPLACEMENTS: dict[str, str] = {"Undefined": "SC"}

# Highest value seen in a meaningful number of bookings, just before a clear gap
# in the frequency table (e.g. babies: 0, 1, 2, then 9 and 10). Above = entry error.
MAX_PLAUSIBLE_COUNTS: dict[str, int] = {
    "adults": 4,
    "children": 3,
    "babies": 2,
    "required_car_parking_spaces": 3,
}

# ADR above this quantile is capped (winsorised) to the quantile value.
ADR_CAP_QUANTILE: float = 0.999

# ---------------------------------------------------------------------------
# Feature engineering (Task 2; reasoning in docs/decisions.md, section 2)
# ---------------------------------------------------------------------------
# Both hotels are in Portugal, so PRT = domestic guest.
DOMESTIC_COUNTRY: str = "PRT"

# Meteorological seasons (northern hemisphere), keyed by month number.
SEASON_BY_MONTH: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}  # fmt: skip

ENGINEERED_FEATURES: list[str] = [
    "total_nights",
    "total_guests",
    "has_children",
    "adr_per_person",
    "stay_value",
    "arrival_month_num",
    "arrival_weekday",
    "arrival_season",
    "is_domestic",
    "previous_cancel_ratio",
]

# ---------------------------------------------------------------------------
# Candidate inputs and feature selection (Task 2)
# ---------------------------------------------------------------------------
# One-hot categories seen in < 1% of training rows are merged into "infrequent".
MIN_CATEGORY_FREQUENCY: float = 0.01
# log1p is applied when a column's skew is above this ...
LOG_SKEW_THRESHOLD: float = 1.0
# ... and log1p shrinks |skew| to at most this fraction of the original.
LOG_SKEW_MAX_RATIO: float = 0.5

NUMERIC_FEATURES: list[str] = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "adr",
    "adr_per_person",
    "stay_value",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "previous_cancel_ratio",
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "arrival_month_num",
    "arrival_weekday",
]
BINARY_FEATURES: list[str] = [
    "is_repeated_guest",
    "has_company",
    "has_agent",
    "has_children",
    "is_domestic",
]
CATEGORICAL_FEATURES: list[str] = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "arrival_date_month",
    "arrival_season",
    "country",
    "agent",
]

# Checks run on the training rows; they FLAG columns, the decisions below are final.
LEAKAGE_PURITY_THRESHOLD: float = 0.99  # one column alone predicts the target
CORRELATION_THRESHOLD: float = 0.9  # |r| above this = redundant pair
NEAR_CONSTANT_THRESHOLD: float = 0.98  # one value in >= 98% of rows

# Candidate features removed from the model input, with the reason (Task 2 decisions,
# re-checked here with the leakage ablation in docs/decisions.md section 3).
DROPPED_FEATURES: dict[str, str] = {
    "assigned_room_type": "Leakage: set at check-in; changed rooms cancel 4.7% vs 31.5%",
    "booking_changes": "Leakage risk: snapshot taken at the outcome date",
    "arrival_date_week_number": "Redundant: r = 0.995 with arrival_month_num",
    "stays_in_week_nights": "Redundant: r = 0.95 with total_nights (= total - weekend)",
    "arrival_date_month": "Redundant: same information as arrival_month_num",
    "arrival_date_year": "Does not generalise: future bookings have unseen years",
    "babies": "Near-constant (98.9% zero); covered by has_children and total_guests",
}

# Leakage ablation: CV score of the reference model with these input changes.
# Each entry: label -> (columns to add, columns to remove).
ABLATION_VARIANTS: dict[str, tuple[list[str], list[str]]] = {
    "Selected features (final)": ([], []),
    "+ booking_changes": (["booking_changes"], []),
    "+ assigned_room_type": (["assigned_room_type"], []),
    "+ both post-booking columns": (["booking_changes", "assigned_room_type"], []),
    "- days_in_waiting_list": ([], ["days_in_waiting_list"]),
    "- country & is_domestic": ([], ["country", "is_domestic"]),
}

# ---------------------------------------------------------------------------
# Validation setup (docs/decisions.md, section 4)
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
# Parallel jobs for CV / search loops (-1 = all cores).
N_JOBS: int = -1

# Temporal check: train on earlier arrivals, test on the latest share of arrivals.
TEMPORAL_TEST_SHARE: float = 0.2

# Metrics computed everywhere. PR-AUC (average precision) is the primary metric:
# it focuses on the minority "canceled" class and does not depend on a threshold.
PRIMARY_METRIC: str = "average_precision"
METRICS: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
}
DEFAULT_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Models (docs/decisions.md, section 5)
# ---------------------------------------------------------------------------
BASELINE_MODEL: str = "dummy"
MODEL_NAMES: dict[str, str] = {
    "dummy": "Dummy (most frequent)",
    "logreg": "Logistic Regression",
    "tree": "Decision Tree",
    "rf": "Random Forest",
    "hgb": "HistGradientBoosting",
    "knn": "K-Nearest Neighbors",
}
TUNED_SUFFIX: str = "_tuned"

# Starting hyperparameters (sensible defaults, before tuning). The Decision Tree is
# left unrestricted on purpose: it shows what an unconstrained tree does (overfits).
MODEL_PARAMS: dict[str, dict[str, object]] = {
    "dummy": {"strategy": "most_frequent"},
    "logreg": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
    "tree": {"max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced"},
    "rf": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    "hgb": {
        "max_iter": 300,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "early_stopping": False,
        "class_weight": "balanced",
    },
    "knn": {"n_neighbors": 25, "weights": "distance"},
}

# KNN predicts by comparing to every stored training row, so it is trained and
# cross-validated on a stratified sample of the training set to keep it practical.
KNN_SAMPLE_SIZE: int = 20_000

# Hyperparameter search (RandomizedSearchCV) on the top models by CV score.
N_MODELS_TO_TUNE: int = 2
TUNING_ITERATIONS: dict[str, int] = {"logreg": 12, "tree": 20, "rf": 12, "hgb": 20, "knn": 8}
PARAM_DISTRIBUTIONS: dict[str, dict[str, list[object]]] = {
    "logreg": {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    },
    "tree": {
        "clf__max_depth": [4, 6, 8, 10, 12, 15, 20, None],
        "clf__min_samples_leaf": [1, 5, 10, 20, 50, 100],
        "clf__criterion": ["gini", "entropy"],
    },
    "rf": {
        "clf__n_estimators": [150, 250, 350],
        "clf__max_depth": [12, 18, 24, 30, None],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", 0.2, 0.35],
    },
    "hgb": {
        "clf__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "clf__max_iter": [200, 300, 500, 700],
        "clf__max_leaf_nodes": [15, 31, 63, 127],
        "clf__min_samples_leaf": [10, 20, 40, 80],
        "clf__l2_regularization": [0.0, 0.1, 1.0, 5.0],
    },
    "knn": {
        "clf__n_neighbors": [5, 11, 25, 41, 61],
        "clf__weights": ["uniform", "distance"],
    },
}

# ---------------------------------------------------------------------------
# Over/underfitting diagnosis and model selection (docs/decisions.md, section 7)
# ---------------------------------------------------------------------------
# Train - CV gap (primary metric) above this = overfitting.
OVERFIT_GAP: float = 0.05
# CV score this far below the best model (with a small gap) = underfitting.
UNDERFIT_MARGIN: float = 0.05
# Models within this CV distance of the best are treated as tied; the tie is broken
# by generalisation gap, then model size, then prediction speed.
SELECTION_TOLERANCE: float = 0.005
# A final model above this size (MB, compressed joblib) is too large for the free host.
MAX_DEPLOY_MODEL_MB: float = 50.0

LEARNING_CURVE_SIZES: list[float] = [0.1, 0.25, 0.5, 0.75, 1.0]
LEARNING_CURVE_MODELS: list[str] = ["logreg", "tree", "rf", "hgb"]
VALIDATION_DEPTHS: list[int] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40]

# ---------------------------------------------------------------------------
# Threshold tuning (docs/decisions.md, section 8)
# ---------------------------------------------------------------------------
THRESHOLD_GRID_START: float = 0.05
THRESHOLD_GRID_STOP: float = 0.95
THRESHOLD_GRID_STEP: float = 0.01
# "max_f1" (balance precision and recall) or "recall_target" (catch a share of cancels).
THRESHOLD_STRATEGY: str = "max_f1"
RECALL_TARGET: float = 0.80
# Business-impact calculator defaults (EUR per booking).
EMPTY_ROOM_COST: float = 250.0  # missed cancellation -> room stays empty
OVERBOOKING_COST: float = 400.0  # false alarm -> overbooked guest has to be relocated

# ---------------------------------------------------------------------------
# Explainability (docs/decisions.md, section 9)
# ---------------------------------------------------------------------------
PERMUTATION_REPEATS: int = 5
TOP_FEATURES_TO_SHOW: int = 15

# ---------------------------------------------------------------------------
# Figures and palette (shared by matplotlib figures and the web app)
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 150
PALETTE: dict[str, str] = {
    "bg_dark": "#0B1020",
    "bg_light": "#F7F8FC",
    "surface_dark": "#141A33",
    "surface_light": "#FFFFFF",
    "primary": "#4F7CFF",  # electric blue
    "pink": "#FF3D81",  # canceled class, risk
    "marigold": "#FFB020",  # warnings, selected model, key numbers
    "teal": "#22D3B6",  # not-canceled class, good metrics
    "violet": "#9B6BFF",
    # Deeper steps of marigold / teal for marks on LIGHT surfaces: the bright brand
    # values have < 3:1 contrast on white (checked with the dataviz palette validator).
    "marigold_deep": "#C98400",
    "teal_deep": "#0E9682",
    "ink": "#141A33",
    "ink_secondary": "#4A5275",
    "ink_muted": "#8A93B2",
    "grid": "#E3E6F0",
}
# Figures are drawn on a light surface, so they use the deeper steps.
CLASS_COLORS: dict[int, str] = {0: PALETTE["teal_deep"], 1: PALETTE["pink"]}
MODEL_COLORS: dict[str, str] = {
    "dummy": PALETTE["ink_muted"],
    "logreg": PALETTE["primary"],
    "tree": PALETTE["marigold_deep"],
    "rf": PALETTE["teal_deep"],
    "hgb": PALETTE["pink"],
    "knn": PALETTE["violet"],
}
# Same models on the dark theme: the bright brand values (validated on #141A33).
MODEL_COLORS_DARK: dict[str, str] = {
    **MODEL_COLORS,
    "tree": PALETTE["marigold"],
    "rf": PALETTE["teal"],
}
# A base model whose tuned version exists is drawn in a lighter tint of its colour.
UNTUNED_TINT: float = 0.5

# ---------------------------------------------------------------------------
# Web app cache (precomputed by main.py, read by the Flask app)
# ---------------------------------------------------------------------------
APP_DIR: Path = PROJECT_ROOT / "app"
APP_CACHE_DIR: Path = APP_DIR / "cache"
CURVE_POINTS: int = 120  # ROC / PR curves are downsampled to this many points
TREE_EXPORT_DEPTH: int = 4  # interactive decision-tree view
FALLING_TREE_DEPTH: int = 3  # "falling bookings" animation
FALLING_SAMPLES: int = 160
PCA_SAMPLES: int = 2_000
REPLAY_MAX_POINTS: int = 150
REPLAY_VALIDATION_SHARE: float = 0.2
APP_RANDOM_SAMPLES: int = 300  # test bookings offered by "try a random booking"
PLAYGROUND_SAMPLE: int = 4_000
PLAYGROUND_POINTS: int = 350
PLAYGROUND_GRID: int = 60
PLAYGROUND_PAIRS: list[tuple[str, str]] = [
    ("lead_time", "adr"),
    ("lead_time", "total_of_special_requests"),
    ("adr", "total_nights"),
]
PLAYGROUND_QUANTILES: tuple[float, float] = (0.01, 0.99)
SWEEP_POINTS: int = 25  # what-if curves on the Predict page

# Predict form: numeric field -> (min, max, step) accepted by the API.
NUMERIC_INPUT_RANGES: dict[str, tuple[float, float, float]] = {
    "lead_time": (0, 700, 1),
    "adr": (0, 500, 1),
    "stays_in_weekend_nights": (0, 10, 1),
    "stays_in_week_nights": (0, 30, 1),
    "adults": (1, 4, 1),
    "children": (0, 3, 1),
    "babies": (0, 2, 1),
    "previous_cancellations": (0, 30, 1),
    "previous_bookings_not_canceled": (0, 80, 1),
    "total_of_special_requests": (0, 5, 1),
    "required_car_parking_spaces": (0, 3, 1),
    "days_in_waiting_list": (0, 400, 1),
}
# Most common categories offered in the form's select boxes (rest -> "Other").
FORM_TOP_COUNTRIES: int = 12
FORM_TOP_AGENTS: int = 10
OTHER_OPTION: str = "Other"
# An agent ID never seen in training: the encoder maps it to "infrequent" (= other agent).
OTHER_AGENT_ID: int = -1
# Categories with fewer training rows than this are not offered in the form.
FORM_MIN_CATEGORY_ROWS: int = 50
# What-if curves: field -> (start, stop) swept on the Predict page.
SWEEP_RANGES: dict[str, tuple[float, float]] = {"lead_time": (0, 500), "adr": (0, 300)}
# Risk bands on the gauge: probability below LOW is "low", above the threshold is "high".
RISK_LOW: float = 0.25
APP_PREDICT_CACHE: Path = APP_CACHE_DIR / "predict.json"
APP_DASHBOARD_CACHE: Path = APP_CACHE_DIR / "dashboard.json"
APP_EXPLORE_CACHE: Path = APP_CACHE_DIR / "explore.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the project's standard format.

    Args:
        level: Minimum log level to display.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, force=True)
