"""Tests for the training workflow in src.train (on a small real sample)."""

import pandas as pd
import pytest

from src import config, train
from src.features import FeatureGroups
from src.models import build_pipeline


@pytest.fixture
def small_cv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use 3 folds and 1 job so the tests stay quick and deterministic."""
    monkeypatch.setattr(config, "CV_FOLDS", 3)
    monkeypatch.setattr(config, "N_JOBS", 1)


def _result(
    key: str, cv: float, train_score: float, size: float, speed: float = 1.0
) -> train.ModelResult:
    """Build a fake ModelResult with the given primary-metric numbers."""
    metric = config.PRIMARY_METRIC
    result = train.ModelResult(
        key=key,
        pipeline=None,  # type: ignore[arg-type]
        cv={m: {"mean": cv, "std": 0.0} for m in config.METRICS},
        train={m: train_score for m in config.METRICS},
        cv_fit_time_s=1.0,
        params={},
        training_rows=100,
    )
    result.size_mb, result.predict_ms_per_1k = size, speed
    assert result.cv[metric]["mean"] == cv
    return result


def test_stratified_sample_keeps_class_balance(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    """The KNN sample keeps the cancel rate; None keeps everything."""
    x, y = xy
    xs, ys = train.stratified_sample(x, y, 1000)
    assert len(xs) == 1000 and abs(ys.mean() - y.mean()) < 0.01
    assert len(train.stratified_sample(x, y, None)[0]) == len(x)


def test_cross_validate_model_records_train_and_cv(
    small_cv: None, groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """CV summary has every metric, train scores and a timing."""
    result = train.cross_validate_model("tree", build_pipeline("tree", groups), *xy)
    assert set(result.cv) == set(config.METRICS)
    assert result.train[config.PRIMARY_METRIC] > result.cv_score  # unlimited tree overfits
    assert result.cv_fit_time_s > 0


def test_tune_model_returns_fitted_tuned_result(
    small_cv: None, groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """The tuned result is fitted, keeps best params and logs every candidate."""
    result = train.tune_model("tree", groups, *xy, n_iter=3)
    assert result.key == "tree_tuned" and result.fitted is not None
    assert len(result.search["candidates"]) == 3
    assert set(result.search["best_params"]) <= {"max_depth", "min_samples_leaf", "criterion"}


def test_rank_excludes_baseline() -> None:
    """Ranking is by CV score and never includes the dummy baseline."""
    results = {
        "dummy": _result("dummy", 0.9, 0.9, 0.1),
        "logreg": _result("logreg", 0.7, 0.7, 1),
        "hgb": _result("hgb", 0.8, 0.8, 1),
    }
    assert train.rank_models(results) == ["hgb", "logreg"]


def test_selection_prefers_smaller_gap_among_ties() -> None:
    """Within tolerance the better-generalising model wins over a slightly higher CV."""
    results = {
        "rf": _result("rf", 0.800, 1.000, 60.0),
        "hgb": _result("hgb", 0.798, 0.830, 1.0),
        "logreg": _result("logreg", 0.700, 0.700, 0.1),
    }
    selection = train.select_final_model(results, tolerance=0.005, max_size_mb=50)
    assert selection["key"] == "hgb"
    assert selection["tied"] == ["rf", "hgb"] and selection["excluded_for_size"] == ["rf"]


def test_selection_keeps_clear_winner() -> None:
    """A model far ahead on CV is chosen even with a larger gap."""
    results = {"rf": _result("rf", 0.85, 0.95, 1.0), "knn": _result("knn", 0.75, 0.76, 1.0)}
    assert train.select_final_model(results)["key"] == "rf"


def test_out_of_fold_proba_covers_every_row(
    small_cv: None, groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Every training row gets exactly one out-of-fold probability."""
    x, y = xy
    proba = train.out_of_fold_proba(build_pipeline("logreg", groups), x, y)
    assert proba.shape == (len(x),) and ((proba >= 0) & (proba <= 1)).all()


def test_depth_validation_curve_shape(
    small_cv: None,
    monkeypatch: pytest.MonkeyPatch,
    groups: FeatureGroups,
    xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    """One train/CV score per depth and the best depth is one of them."""
    monkeypatch.setattr(config, "VALIDATION_DEPTHS", [1, 3, 8])
    curve = train.depth_validation_curve(groups, *xy)
    assert len(curve["cv_mean"]) == 3 and curve["best_depth"] in [1, 3, 8]


def test_one_standard_error_rule_prefers_smaller_gap() -> None:
    """Within one std of the best CV score, the least-overfitting setting wins."""
    metric = config.PRIMARY_METRIC
    cv_results = {
        f"mean_test_{metric}": [0.800, 0.797, 0.780],
        f"std_test_{metric}": [0.004, 0.004, 0.004],
        f"mean_train_{metric}": [0.950, 0.850, 0.790],
    }
    assert train.one_standard_error_index(cv_results) == 1
