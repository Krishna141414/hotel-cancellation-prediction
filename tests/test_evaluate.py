"""Tests for metrics, diagnosis and threshold helpers in src.evaluate."""

import numpy as np
import pandas as pd
import pytest

from src import config, evaluate
from src.features import FeatureGroups
from src.models import build_pipeline

Y = np.array([0, 0, 0, 0, 1, 1, 0, 1])
P = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9, 0.4, 0.35])


def test_compute_metrics_known_values() -> None:
    """Hand-checked metrics at threshold 0.5: TP=2, FP=1, FN=1, TN=4."""
    metrics = evaluate.compute_metrics(Y, P)
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert set(metrics) == set(config.METRICS)


def test_confusion_layout() -> None:
    """Matrix is [[TN, FP], [FN, TP]]."""
    assert evaluate.confusion(Y, P) == [[4, 1], [1, 2]]
    assert evaluate.confusion(Y, P, threshold=0.3) == [[2, 3], [0, 3]]


def test_baseline_metrics_do_not_warn() -> None:
    """A model that never predicts 'canceled' gets precision/recall 0, not an error."""
    metrics = evaluate.compute_metrics(Y, np.zeros_like(P))
    assert metrics["precision"] == 0 and metrics["recall"] == 0


def test_curve_points_are_downsampled_and_bounded() -> None:
    """Curves have at most the configured points, all within [0, 1]."""
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 5000), rng.random(5000)
    curves = evaluate.curve_points(y, p, max_points=50)
    for kind in ("roc", "pr"):
        points = np.asarray(curves[kind])
        assert len(points) <= 50 and points.min() >= 0 and points.max() <= 1
    assert curves["roc"][0] == [0.0, 0.0] and curves["roc"][-1] == [1.0, 1.0]


@pytest.mark.parametrize(
    ("key", "train", "cv", "best", "label"),
    [
        ("dummy", 0.27, 0.27, 0.8, "Baseline"),
        ("tree", 1.0, 0.5, 0.8, "Strong overfitting"),
        ("rf", 0.87, 0.80, 0.8, "Overfitting"),
        ("logreg", 0.69, 0.68, 0.8, "Underfitting"),
        ("hgb", 0.82, 0.79, 0.8, "Good fit"),
    ],
)
def test_diagnose_fit(key: str, train: float, cv: float, best: float, label: str) -> None:
    """Each rule produces the expected label."""
    assert evaluate.diagnose_fit(key, train, cv, best)["label"] == label


def test_threshold_table_and_choice() -> None:
    """Counts add up at every threshold and both strategies return a grid value."""
    table = evaluate.threshold_table(Y, P)
    assert ((table[["tp", "fp", "fn", "tn"]].sum(axis=1)) == len(Y)).all()
    assert evaluate.choose_threshold(table, "max_f1") in table["threshold"].tolist()
    recall_t = evaluate.choose_threshold(table, "recall_target", 1.0)
    assert table.set_index("threshold").loc[recall_t, "recall"] == 1.0
    with pytest.raises(ValueError, match="Unknown"):
        evaluate.choose_threshold(table, "magic")


def test_best_cost_threshold_follows_costs() -> None:
    """Expensive false alarms push the best threshold up; savings are vs 'no model'."""
    table = evaluate.threshold_table(Y, P)
    cheap_alarms = evaluate.best_cost_threshold(table, empty_room_cost=100, overbooking_cost=1)
    costly_alarms = evaluate.best_cost_threshold(table, empty_room_cost=1, overbooking_cost=100)
    assert cheap_alarms["threshold"] <= costly_alarms["threshold"]
    assert cheap_alarms["baseline_cost"] == 3 * 100
    assert cheap_alarms["savings"] >= 0


def test_importance_tables(groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    """Permutation importance covers every input; tree importances sum to 1."""
    x, y = xy
    model = build_pipeline("tree", groups, {"max_depth": 4}).fit(x, y)
    permutation = evaluate.permutation_importance_table(model, x.head(500), y.head(500), 2)
    assert set(permutation["feature"]) == set(groups.all)
    tree = evaluate.tree_importance_table(model, groups)
    assert tree["importance"].sum() == pytest.approx(1.0, abs=1e-3)
    assert set(tree["feature"]) <= set(groups.all)
    with pytest.raises(ValueError, match="feature_importances_"):
        evaluate.tree_importance_table(build_pipeline("logreg", groups).fit(x, y), groups)
