"""Training workflow: cross-validation, tuning, model selection and final fit.

The test set is never used here. Every decision (ranking, tuning, selection,
threshold) is made from cross-validation on the training set; ``main.py``
evaluates on the test set once, after the final model is frozen.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline

from src import config
from src.data import clean_data, load_raw_data
from src.evaluate import compute_metrics, make_scorers
from src.features import (
    FeatureGroups,
    arrival_dates,
    engineer_features,
    split_frame,
    temporal_split,
)
from src.models import MODEL_SPECS, base_key, build_pipeline, display_name, model_size_mb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """Cleaned + engineered bookings and the fixed stratified train/test split."""

    featured: pd.DataFrame
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    groups: FeatureGroups
    adr_cap: float
    raw_shape: tuple[int, int]
    cleaning_report: pd.DataFrame

    @property
    def x_train(self) -> pd.DataFrame:
        """Training inputs."""
        return self.train_rows[self.groups.all]

    @property
    def y_train(self) -> pd.Series:
        """Training labels."""
        return self.train_rows[config.TARGET]

    @property
    def x_test(self) -> pd.DataFrame:
        """Test inputs (used once, for the final report)."""
        return self.test_rows[self.groups.all]

    @property
    def y_test(self) -> pd.Series:
        """Test labels."""
        return self.test_rows[config.TARGET]


def prepare_dataset(raw: pd.DataFrame | None = None) -> Dataset:
    """Load, clean (Task 2 rules), engineer features and split train/test.

    Args:
        raw: Raw bookings; read from ``config.RAW_DATA_PATH`` when ``None``.

    Returns:
        The prepared :class:`Dataset`.
    """
    raw = load_raw_data() if raw is None else raw
    cleaning = clean_data(raw)
    featured = engineer_features(cleaning.data)
    train_rows, test_rows = split_frame(featured)
    groups = FeatureGroups.selected()
    logger.info(
        "Split: %s train / %s test rows (stratified, cancel rate %.1f%% / %.1f%%); %d inputs",
        f"{len(train_rows):,}",
        f"{len(test_rows):,}",
        train_rows[config.TARGET].mean() * 100,
        test_rows[config.TARGET].mean() * 100,
        len(groups.all),
    )
    return Dataset(
        featured=featured,
        train_rows=train_rows,
        test_rows=test_rows,
        groups=groups,
        adr_cap=cleaning.adr_cap,
        raw_shape=(len(raw), raw.shape[1]),
        cleaning_report=cleaning.report,
    )


def cv_splitter(folds: int | None = None) -> StratifiedKFold:
    """Stratified, shuffled K-fold with the project seed (same folds for every model).

    Args:
        folds: Number of folds (default ``config.CV_FOLDS``, read at call time).

    Returns:
        The splitter.
    """
    n_splits = folds or config.CV_FOLDS
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def stratified_sample(
    x: pd.DataFrame, y: pd.Series, size: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a stratified sample of ``size`` rows (or the data unchanged).

    Args:
        x: Features.
        y: Labels.
        size: Rows to keep; ``None`` or >= ``len(x)`` keeps everything.

    Returns:
        ``(x_sample, y_sample)``.
    """
    if size is None or size >= len(x):
        return x, y
    x_sample, _, y_sample, _ = train_test_split(
        x, y, train_size=size, stratify=y, random_state=config.RANDOM_STATE
    )
    return x_sample, y_sample


def training_data_for(key: str, x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Training rows used for a model (KNN uses a stratified sample).

    Args:
        key: Model key.
        x: Full training features.
        y: Full training labels.

    Returns:
        ``(x, y)`` for that model.
    """
    return stratified_sample(x, y, MODEL_SPECS[base_key(key)].sample_size)


# ---------------------------------------------------------------------------
# Cross-validated comparison
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    """Everything known about one model (cross-validation first, test added later)."""

    key: str
    pipeline: Pipeline
    cv: dict[str, dict[str, float]]
    train: dict[str, float]
    cv_fit_time_s: float
    params: dict[str, Any]
    training_rows: int
    search: dict[str, Any] | None = None
    fitted: Pipeline | None = None
    fit_time_s: float | None = None
    predict_ms_per_1k: float | None = None
    size_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Display name."""
        return display_name(self.key)

    @property
    def cv_score(self) -> float:
        """Mean CV score on the primary metric."""
        return self.cv[config.PRIMARY_METRIC]["mean"]

    @property
    def gap(self) -> float:
        """Train - CV gap on the primary metric (overfitting signal)."""
        return round(self.train[config.PRIMARY_METRIC] - self.cv_score, 4)


def _summarise_scores(scores: dict[str, np.ndarray], prefix: str) -> dict[str, dict[str, float]]:
    """Mean and std per metric from ``cross_validate``-style arrays.

    Args:
        scores: Arrays keyed like ``"test_roc_auc"``.
        prefix: ``"test"`` or ``"train"``.

    Returns:
        Metric -> ``{"mean", "std"}``.
    """
    return {
        metric: {
            "mean": round(float(np.mean(scores[f"{prefix}_{metric}"])), 4),
            "std": round(float(np.std(scores[f"{prefix}_{metric}"])), 4),
        }
        for metric in config.METRICS
    }


def cross_validate_model(
    key: str, pipeline: Pipeline, x: pd.DataFrame, y: pd.Series
) -> ModelResult:
    """5-fold stratified CV with train-fold scores and timings.

    Args:
        key: Model key.
        pipeline: Unfitted model pipeline.
        x: Training features.
        y: Training labels.

    Returns:
        The model's :class:`ModelResult` (not yet fitted on the full train set).
    """
    x_used, y_used = training_data_for(key, x, y)
    scores = cross_validate(
        pipeline,
        x_used,
        y_used,
        cv=cv_splitter(),
        scoring=make_scorers(),
        return_train_score=True,
        n_jobs=config.N_JOBS,
    )
    cv = _summarise_scores(scores, "test")
    train = {
        metric: values["mean"] for metric, values in _summarise_scores(scores, "train").items()
    }
    result = ModelResult(
        key=key,
        pipeline=pipeline,
        cv=cv,
        train=train,
        cv_fit_time_s=round(float(np.mean(scores["fit_time"])), 3),
        params=dict(config.MODEL_PARAMS[base_key(key)]),
        training_rows=len(x_used),
    )
    logger.info(
        "  %-30s CV %s %.4f +/- %.4f | train %.4f | fold fit %.1fs",
        result.name,
        config.METRIC_LABELS[config.PRIMARY_METRIC],
        result.cv_score,
        cv[config.PRIMARY_METRIC]["std"],
        train[config.PRIMARY_METRIC],
        result.cv_fit_time_s,
    )
    return result


def run_model_comparison(
    zoo: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series
) -> dict[str, ModelResult]:
    """Cross-validate every model in the zoo on the same folds.

    Args:
        zoo: Model key -> unfitted pipeline.
        x: Training features.
        y: Training labels.

    Returns:
        Model key -> result.
    """
    logger.info("Cross-validating %d models (%d-fold stratified)", len(zoo), config.CV_FOLDS)
    return {key: cross_validate_model(key, pipeline, x, y) for key, pipeline in zoo.items()}


def rank_models(results: dict[str, ModelResult], metric: str = config.PRIMARY_METRIC) -> list[str]:
    """Model keys sorted by mean CV score, best first (baseline excluded).

    Args:
        results: Model results.
        metric: Metric to rank by.

    Returns:
        Ranked keys.
    """
    candidates = [key for key in results if key != config.BASELINE_MODEL]
    return sorted(candidates, key=lambda key: results[key].cv[metric]["mean"], reverse=True)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_model(
    key: str,
    groups: FeatureGroups,
    x: pd.DataFrame,
    y: pd.Series,
    n_iter: int | None = None,
) -> ModelResult:
    """RandomizedSearchCV over ``config.PARAM_DISTRIBUTIONS`` on the same CV folds.

    Every metric is recorded for every candidate; the best candidate is chosen on
    the primary metric and refitted on the full training set.

    Args:
        key: Base algorithm key.
        groups: Model input columns.
        x: Training features.
        y: Training labels.
        n_iter: Number of sampled settings (default ``config.TUNING_ITERATIONS``).

    Returns:
        Result for ``"<key>_tuned"``, already fitted on the training rows.
    """
    x_used, y_used = training_data_for(key, x, y)
    n_iter = n_iter or config.TUNING_ITERATIONS[key]
    search = RandomizedSearchCV(
        build_pipeline(key, groups),
        config.PARAM_DISTRIBUTIONS[key],
        n_iter=n_iter,
        scoring=make_scorers(),
        refit=config.PRIMARY_METRIC,
        cv=cv_splitter(),
        return_train_score=True,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
    )
    started = time.perf_counter()
    search.fit(x_used, y_used)
    elapsed = time.perf_counter() - started
    best = search.best_index_
    cv_results = search.cv_results_
    cv = {
        metric: {
            "mean": round(float(cv_results[f"mean_test_{metric}"][best]), 4),
            "std": round(float(cv_results[f"std_test_{metric}"][best]), 4),
        }
        for metric in config.METRICS
    }
    train = {
        metric: round(float(cv_results[f"mean_train_{metric}"][best]), 4)
        for metric in config.METRICS
    }
    best_params = {name.removeprefix("clf__"): value for name, value in search.best_params_.items()}
    tuned_key = f"{key}{config.TUNED_SUFFIX}"
    candidates = sorted(
        (
            {
                "params": {n.removeprefix("clf__"): v for n, v in cv_results["params"][i].items()},
                "cv": round(float(cv_results[f"mean_test_{config.PRIMARY_METRIC}"][i]), 4),
                "train": round(float(cv_results[f"mean_train_{config.PRIMARY_METRIC}"][i]), 4),
            }
            for i in range(len(cv_results["params"]))
        ),
        key=lambda row: row["cv"],
        reverse=True,
    )
    result = ModelResult(
        key=tuned_key,
        pipeline=build_pipeline(key, groups, best_params),
        cv=cv,
        train=train,
        cv_fit_time_s=round(float(cv_results["mean_fit_time"][best]), 3),
        params={**config.MODEL_PARAMS[key], **best_params},
        training_rows=len(x_used),
        search={
            "n_iter": n_iter,
            "folds": config.CV_FOLDS,
            "fits": n_iter * config.CV_FOLDS,
            "search_time_s": round(elapsed, 1),
            "best_params": best_params,
            "candidates": candidates,
        },
        fitted=search.best_estimator_,
        fit_time_s=round(float(search.refit_time_), 3),
    )
    logger.info(
        "  %-30s best CV %.4f (%d settings x %d folds in %.0fs): %s",
        result.name,
        result.cv_score,
        n_iter,
        config.CV_FOLDS,
        elapsed,
        best_params,
    )
    return result


# ---------------------------------------------------------------------------
# Full-train fit, timing and size
# ---------------------------------------------------------------------------
def fit_and_profile(result: ModelResult, x: pd.DataFrame, y: pd.Series) -> ModelResult:
    """Fit on the full training set (if not yet fitted) and record time, speed and size.

    Prediction speed is measured on training rows, so the test set stays untouched.

    Args:
        result: Model result from CV or tuning.
        x: Training features.
        y: Training labels.

    Returns:
        The same result with ``fitted``, ``fit_time_s``, ``predict_ms_per_1k`` and ``size_mb``.
    """
    if result.fitted is None:
        x_used, y_used = training_data_for(result.key, x, y)
        started = time.perf_counter()
        result.fitted = clone(result.pipeline).fit(x_used, y_used)
        result.fit_time_s = round(time.perf_counter() - started, 3)
    timing_rows = x.iloc[:10_000]
    started = time.perf_counter()
    result.fitted.predict_proba(timing_rows)
    result.predict_ms_per_1k = round(
        (time.perf_counter() - started) / len(timing_rows) * 1000 * 1000, 2
    )
    result.size_mb = model_size_mb(result.fitted)
    logger.info(
        "  %-30s fit %.1fs | predict %.1f ms / 1k rows | %.2f MB",
        result.name,
        result.fit_time_s,
        result.predict_ms_per_1k,
        result.size_mb,
    )
    return result


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def select_final_model(
    results: dict[str, ModelResult],
    tolerance: float = config.SELECTION_TOLERANCE,
    max_size_mb: float = config.MAX_DEPLOY_MODEL_MB,
) -> dict[str, Any]:
    """Choose the final model from CV evidence (never from test scores).

    1. Candidates = models within ``tolerance`` of the best mean CV score (differences
       that small are within fold-to-fold noise).
    2. Drop candidates too large to deploy on the free host (if any remain).
    3. Among the rest, pick the smallest train - CV gap (generalises best), then the
       smallest file, then the fastest prediction.

    Args:
        results: Fitted and profiled model results (baseline is ignored).
        tolerance: CV distance from the best that counts as a tie.
        max_size_mb: Largest deployable model.

    Returns:
        ``{"key", "best_cv_key", "tied", "excluded_for_size", "rule"}``.
    """
    ranked = rank_models(results)
    best_cv = results[ranked[0]].cv_score
    tied = [key for key in ranked if results[key].cv_score >= best_cv - tolerance]
    deployable = [key for key in tied if (results[key].size_mb or 0) <= max_size_mb]
    excluded = [key for key in tied if key not in deployable]
    pool = deployable or tied
    chosen = min(
        pool,
        key=lambda key: (
            results[key].gap,
            results[key].size_mb or 0,
            results[key].predict_ms_per_1k or 0,
        ),
    )
    logger.info(
        "Selection: best CV = %s; tied within %.3f: %s; too large: %s -> %s",
        results[ranked[0]].name,
        tolerance,
        [results[k].name for k in tied],
        [results[k].name for k in excluded],
        results[chosen].name,
    )
    return {
        "key": chosen,
        "best_cv_key": ranked[0],
        "ranking": ranked,
        "tied": tied,
        "excluded_for_size": excluded,
        "rule": (
            f"Models within {tolerance} {config.METRIC_LABELS[config.PRIMARY_METRIC]} of the best "
            f"CV score are tied; models above {max_size_mb:.0f} MB are excluded; the tie is "
            "broken by the smallest train-CV gap, then file size, then prediction speed."
        ),
    }


# ---------------------------------------------------------------------------
# Threshold, learning and validation curves (training data only)
# ---------------------------------------------------------------------------
def out_of_fold_proba(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Cross-validated probabilities for every training row (for threshold tuning).

    Args:
        pipeline: Unfitted final pipeline.
        x: Training features.
        y: Training labels.

    Returns:
        Probability of "canceled" for each row, predicted by a model that never saw it.
    """
    proba = cross_val_predict(
        clone(pipeline), x, y, cv=cv_splitter(), method="predict_proba", n_jobs=config.N_JOBS
    )
    return proba[:, 1]


def learning_curve_data(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    sizes: list[float] = config.LEARNING_CURVE_SIZES,
) -> dict[str, Any]:
    """Train and CV score as the training set grows.

    Args:
        pipeline: Unfitted pipeline.
        x: Training features.
        y: Training labels.
        sizes: Training-set fractions to evaluate.

    Returns:
        ``{"sizes", "train_mean", "train_std", "cv_mean", "cv_std"}``.
    """
    sizes, train_scores, cv_scores = learning_curve(
        clone(pipeline),
        x,
        y,
        train_sizes=sizes,
        cv=cv_splitter(),
        scoring=config.PRIMARY_METRIC,
        shuffle=True,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
    )
    return {
        "sizes": sizes.tolist(),
        "train_mean": train_scores.mean(axis=1).round(4).tolist(),
        "train_std": train_scores.std(axis=1).round(4).tolist(),
        "cv_mean": cv_scores.mean(axis=1).round(4).tolist(),
        "cv_std": cv_scores.std(axis=1).round(4).tolist(),
    }


def depth_validation_curve(groups: FeatureGroups, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Decision Tree train vs CV score for each ``max_depth`` in the config.

    Args:
        groups: Model input columns.
        x: Training features.
        y: Training labels.

    Returns:
        ``{"depths", "train_mean", "train_std", "cv_mean", "cv_std", "best_depth"}``.
    """
    train_scores, cv_scores = validation_curve(
        build_pipeline("tree", groups),
        x,
        y,
        param_name="clf__max_depth",
        param_range=config.VALIDATION_DEPTHS,
        cv=cv_splitter(),
        scoring=config.PRIMARY_METRIC,
        n_jobs=config.N_JOBS,
    )
    cv_mean = cv_scores.mean(axis=1)
    return {
        "depths": list(config.VALIDATION_DEPTHS),
        "train_mean": train_scores.mean(axis=1).round(4).tolist(),
        "train_std": train_scores.std(axis=1).round(4).tolist(),
        "cv_mean": cv_mean.round(4).tolist(),
        "cv_std": cv_scores.std(axis=1).round(4).tolist(),
        "best_depth": int(config.VALIDATION_DEPTHS[int(np.argmax(cv_mean))]),
    }


# ---------------------------------------------------------------------------
# Extra checks: temporal validation and leakage ablation
# ---------------------------------------------------------------------------
def _fit_and_score(
    key: str, pipeline: Pipeline, train_rows: pd.DataFrame, test_rows: pd.DataFrame, columns: list
) -> tuple[str, dict[str, float]]:
    """Fit a fresh copy of a pipeline on one frame and score it on another.

    Args:
        key: Model key (decides whether a sample is used).
        pipeline: Unfitted pipeline.
        train_rows: Rows to fit on.
        test_rows: Rows to evaluate on.
        columns: Input columns.

    Returns:
        ``(key, metrics)``.
    """
    x_fit, y_fit = training_data_for(key, train_rows[columns], train_rows[config.TARGET])
    model = clone(pipeline).fit(x_fit, y_fit)
    proba = model.predict_proba(test_rows[columns])[:, 1]
    return key, compute_metrics(test_rows[config.TARGET], proba)


def temporal_validation(
    results: dict[str, ModelResult], featured: pd.DataFrame, groups: FeatureGroups
) -> dict[str, Any]:
    """Train on earlier arrivals, test on the latest period (data-drift check).

    This is a separate experiment on all cleaned rows ordered by arrival date; it
    does not feed any decision.

    Args:
        results: Models to check (their configured/tuned pipelines are refitted).
        featured: All cleaned + engineered bookings.
        groups: Model input columns.

    Returns:
        Period info and per-model metrics on the latest period.
    """
    train_rows, test_rows, cutoff = temporal_split(featured)
    logger.info(
        "Temporal split at %s: %s earlier / %s later arrivals (cancel rate %.1f%% / %.1f%%)",
        cutoff.date(),
        f"{len(train_rows):,}",
        f"{len(test_rows):,}",
        train_rows[config.TARGET].mean() * 100,
        test_rows[config.TARGET].mean() * 100,
    )
    scored = Parallel(n_jobs=config.N_JOBS)(
        delayed(_fit_and_score)(key, result.pipeline, train_rows, test_rows, groups.all)
        for key, result in results.items()
    )
    return {
        "cutoff": str(cutoff.date()),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_cancel_rate": round(float(train_rows[config.TARGET].mean()), 4),
        "test_cancel_rate": round(float(test_rows[config.TARGET].mean()), 4),
        "train_period": [
            str(_arrival_bound(train_rows, min)),
            str(_arrival_bound(train_rows, max)),
        ],
        "test_period": [str(cutoff.date()), str(_arrival_bound(test_rows, max))],
        "metrics": dict(scored),
    }


def _arrival_bound(rows: pd.DataFrame, pick: Any) -> object:  # noqa: ANN401
    """Earliest or latest arrival date in ``rows``.

    Args:
        rows: Bookings with arrival-date columns.
        pick: ``min`` or ``max``.

    Returns:
        The date.
    """
    return pick(arrival_dates(rows)).date()


def leakage_ablation(
    key: str, params: dict[str, Any], groups: FeatureGroups, train_rows: pd.DataFrame
) -> list[dict[str, Any]]:
    """CV score of the chosen model with suspected post-booking columns added or removed.

    Evidence for the leakage decisions: if adding a column makes the model much
    better, the column carries outcome information that is not known at booking time.

    Args:
        key: Base algorithm key of the final model.
        params: Its hyperparameters.
        groups: Final model inputs.
        train_rows: Training rows (all candidate columns available).

    Returns:
        One row per variant with CV PR-AUC and ROC-AUC (mean, std) and the change
        versus the final feature set.
    """
    rows = []
    for label, (add, remove) in config.ABLATION_VARIANTS.items():
        variant = groups.with_columns(add).without(remove)
        x_used, y_used = training_data_for(key, train_rows[variant.all], train_rows[config.TARGET])
        scores = cross_validate(
            build_pipeline(key, variant, params),
            x_used,
            y_used,
            cv=cv_splitter(),
            scoring=[config.PRIMARY_METRIC, "roc_auc"],
            n_jobs=config.N_JOBS,
        )
        rows.append(
            {
                "variant": label,
                "inputs": len(variant.all),
                "pr_auc": round(float(scores[f"test_{config.PRIMARY_METRIC}"].mean()), 4),
                "pr_auc_std": round(float(scores[f"test_{config.PRIMARY_METRIC}"].std()), 4),
                "roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
            }
        )
        logger.info("  %-30s CV PR-AUC %.4f", label, rows[-1]["pr_auc"])
    reference = rows[0]["pr_auc"]
    for row in rows:
        row["delta"] = round(row["pr_auc"] - reference, 4)
    return rows
