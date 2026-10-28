"""Metrics, curves, over/underfitting diagnosis, threshold analysis and importances.

Pure functions on true labels and predicted probabilities, so they are easy to
test and are shared by training, the figures and the web-app cache.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src import config
from src.features import FeatureGroups, readable_feature

logger = logging.getLogger(__name__)

# Threshold-based metrics: (y_true, y_pred) -> score. zero_division=0 keeps the
# baseline (which never predicts "canceled") from raising warnings.
_LABEL_METRICS: dict[str, Callable[..., float]] = {
    "accuracy": accuracy_score,
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, zero_division=0),
    "f1": lambda y, p: f1_score(y, p, zero_division=0),
}
# Ranking metrics: (y_true, probability) -> score.
_SCORE_METRICS: dict[str, Callable[..., float]] = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
}


def make_scorers() -> dict[str, Any]:
    """Build sklearn scorers for every metric in ``config.METRICS``.

    Returns:
        Metric name -> scorer usable in ``cross_validate`` / ``RandomizedSearchCV``.
    """
    scorers: dict[str, Any] = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    for name in ("precision", "recall", "f1"):
        scorer_fn = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[name]
        scorers[name] = make_scorer(scorer_fn, zero_division=0)
    return {name: scorers[name] for name in config.METRICS}


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute every metric in ``config.METRICS`` from probabilities.

    Args:
        y_true: True 0/1 labels.
        proba: Predicted probability of class 1 (canceled).
        threshold: Probability at or above which a booking is predicted canceled.

    Returns:
        Metric name -> value (rounded to 4 decimals).
    """
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    metrics = {name: fn(y_true, y_pred) for name, fn in _LABEL_METRICS.items()}
    metrics |= {name: fn(y_true, proba) for name, fn in _SCORE_METRICS.items()}
    return {name: round(float(metrics[name]), 4) for name in config.METRICS}


def confusion(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float = config.DEFAULT_THRESHOLD
) -> list[list[int]]:
    """Confusion matrix ``[[TN, FP], [FN, TP]]`` at a threshold.

    Args:
        y_true: True 0/1 labels.
        proba: Predicted probability of class 1.
        threshold: Decision threshold.

    Returns:
        2 x 2 nested list of counts.
    """
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()


def report_dict(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Sklearn classification report as a dictionary.

    Args:
        y_true: True 0/1 labels.
        proba: Predicted probability of class 1.
        threshold: Decision threshold.

    Returns:
        Per-class precision/recall/F1/support plus averages.
    """
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, target_names=list(config.CLASS_NAMES), output_dict=True, zero_division=0
    )


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Keep at most ``max_points`` rows, always including the first and last.

    Args:
        points: Array of curve points (rows).
        max_points: Maximum rows to keep.

    Returns:
        The downsampled array.
    """
    if len(points) <= max_points:
        return points
    index = np.unique(np.linspace(0, len(points) - 1, max_points).round().astype(int))
    return points[index]


def curve_points(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, max_points: int = config.CURVE_POINTS
) -> dict[str, list[list[float]]]:
    """ROC and precision-recall curves, downsampled for charts.

    Args:
        y_true: True 0/1 labels.
        proba: Predicted probability of class 1.
        max_points: Maximum points per curve.

    Returns:
        ``{"roc": [[fpr, tpr], ...], "pr": [[recall, precision], ...]}``.
    """
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    roc = _downsample(np.column_stack([fpr, tpr]), max_points)
    # Recall decreases along the PR arrays; reverse so x runs left to right.
    pr = _downsample(np.column_stack([recall, precision])[::-1], max_points)
    return {"roc": np.round(roc, 4).tolist(), "pr": np.round(pr, 4).tolist()}


# ---------------------------------------------------------------------------
# Over/underfitting
# ---------------------------------------------------------------------------
def diagnose_fit(
    key: str,
    train_score: float,
    cv_score: float,
    best_cv_score: float,
    overfit_gap: float = config.OVERFIT_GAP,
    underfit_margin: float = config.UNDERFIT_MARGIN,
) -> dict[str, str]:
    """Label a model as baseline, overfitting, underfitting or good fit.

    Rules (primary metric): a train - CV gap above ``overfit_gap`` means the model
    memorises training rows; a small gap but a CV score more than
    ``underfit_margin`` below the best model means it is too simple to capture
    the pattern.

    Args:
        key: Model key (the baseline is labelled separately).
        train_score: Mean score on the CV training folds.
        cv_score: Mean score on the CV validation folds.
        best_cv_score: Best CV score among all models.
        overfit_gap: Gap above which the model is called overfitting.
        underfit_margin: Distance below the best CV score that means underfitting.

    Returns:
        ``{"label": ..., "reason": ...}``.
    """
    gap = train_score - cv_score
    behind = best_cv_score - cv_score
    if key == config.BASELINE_MODEL:
        return {
            "label": "Baseline",
            "reason": "No learning at all: always predicts the majority class.",
        }
    if gap > overfit_gap:
        severity = "Strong overfitting" if gap > 2 * overfit_gap else "Mild overfitting"
        return {
            "label": severity,
            "reason": f"Train score is {gap:.3f} above CV: it memorises training rows "
            f"that do not carry over to unseen bookings.",
        }
    if behind > underfit_margin:
        return {
            "label": "Underfitting",
            "reason": f"Train and CV are close (gap {gap:.3f}) but {behind:.3f} below the best "
            f"model: too simple to capture the non-linear patterns.",
        }
    return {
        "label": "Good fit",
        "reason": f"Small train-CV gap ({gap:.3f}) and close to the best CV score.",
    }


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------
def threshold_grid() -> np.ndarray:
    """Candidate decision thresholds from ``src/config.py``.

    Returns:
        Sorted thresholds.
    """
    grid = np.arange(
        config.THRESHOLD_GRID_START,
        config.THRESHOLD_GRID_STOP + config.THRESHOLD_GRID_STEP / 2,
        config.THRESHOLD_GRID_STEP,
    )
    return np.round(grid, 4)


def threshold_table(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Precision, recall, F1 and confusion counts at every threshold.

    Args:
        y_true: True 0/1 labels.
        proba: Predicted probability of class 1.
        thresholds: Thresholds to evaluate (default :func:`threshold_grid`).

    Returns:
        One row per threshold with ``threshold, precision, recall, f1, tp, fp, fn, tn``.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba)
    rows = []
    for threshold in threshold_grid() if thresholds is None else thresholds:
        pred = p >= threshold
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    table: pd.DataFrame,
    strategy: str = config.THRESHOLD_STRATEGY,
    recall_target: float = config.RECALL_TARGET,
) -> float:
    """Pick the decision threshold from a :func:`threshold_table`.

    Args:
        table: Threshold table (computed on out-of-fold training predictions).
        strategy: ``"max_f1"`` or ``"recall_target"`` (highest threshold that still
            reaches ``recall_target``, i.e. the fewest false alarms for that recall).
        recall_target: Minimum recall for the ``"recall_target"`` strategy.

    Returns:
        The chosen threshold.

    Raises:
        ValueError: If the strategy is unknown or no threshold reaches the target.
    """
    if strategy == "max_f1":
        return float(table.loc[table["f1"].idxmax(), "threshold"])
    if strategy == "recall_target":
        reaching = table[table["recall"] >= recall_target]
        if reaching.empty:
            raise ValueError(f"No threshold reaches recall {recall_target}")
        return float(reaching["threshold"].max())
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def best_cost_threshold(
    table: pd.DataFrame, empty_room_cost: float, overbooking_cost: float
) -> dict[str, float]:
    """Threshold with the lowest total cost of mistakes (business view).

    A missed cancellation (FN) leaves a room empty; a false alarm (FP) leads to an
    overbooked guest who must be relocated.

    Args:
        table: Threshold table with confusion counts.
        empty_room_cost: Cost of one false negative.
        overbooking_cost: Cost of one false positive.

    Returns:
        ``{"threshold", "cost", "baseline_cost", "savings"}`` where the baseline is
        "no model" (never overbook, every cancellation leaves a room empty).
    """
    cost = table["fn"] * empty_room_cost + table["fp"] * overbooking_cost
    best = int(cost.idxmin())
    baseline = float((table["tp"] + table["fn"]).iloc[0] * empty_room_cost)
    return {
        "threshold": float(table.loc[best, "threshold"]),
        "cost": float(cost.iloc[best]),
        "baseline_cost": baseline,
        "savings": baseline - float(cost.iloc[best]),
    }


# ---------------------------------------------------------------------------
# Importances
# ---------------------------------------------------------------------------
def permutation_importance_table(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = config.PERMUTATION_REPEATS,
) -> pd.DataFrame:
    """Drop in the primary metric when each input column is shuffled.

    Shuffling works on the raw input columns (before one-hot), so a categorical
    column is measured as one feature.

    Args:
        model: Fitted pipeline.
        x: Evaluation features.
        y: Evaluation labels.
        n_repeats: Shuffles per column.

    Returns:
        ``feature, importance_mean, importance_std`` sorted by importance.
    """
    result = permutation_importance(
        model,
        x,
        y,
        scoring=config.PRIMARY_METRIC,
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": x.columns,
            "importance_mean": result.importances_mean.round(5),
            "importance_std": result.importances_std.round(5),
        }
    )
    return table.sort_values("importance_mean", ascending=False, ignore_index=True)


def tree_importance_table(model: Pipeline, groups: FeatureGroups) -> pd.DataFrame:
    """Impurity-based importances of a fitted tree model, summed per input column.

    Args:
        model: Fitted pipeline whose ``clf`` has ``feature_importances_``.
        groups: Model input columns.

    Returns:
        ``feature, importance`` (sums to 1) sorted by importance.

    Raises:
        ValueError: If the model has no impurity importances.
    """
    estimator = model.named_steps["clf"]
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError(f"{type(estimator).__name__} has no feature_importances_")
    names = model.named_steps["prep"].get_feature_names_out()
    columns = [readable_feature(name, groups)[0] for name in names]
    table = (
        pd.DataFrame({"feature": columns, "importance": estimator.feature_importances_})
        .groupby("feature", as_index=False)["importance"]
        .sum()
    )
    table["importance"] = table["importance"].round(5)
    return table.sort_values("importance", ascending=False, ignore_index=True)
