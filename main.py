"""Run the whole project: download -> clean -> compare -> tune -> select -> evaluate -> save.

Usage:
    python main.py            # full run (about 15 minutes on a laptop)
    python main.py --quick    # smoke test: tiny tuning budget, fewer curve points
    python main.py --figures-only   # redraw reports/figures from the saved summary

Outputs: ``models/final_model.joblib``, ``models/metrics.json``,
``reports/model_comparison.csv``, ``reports/figures/*.png`` and the web-app cache
in ``app/cache/``.
"""

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.build_cache import build_predict_cache
from src import config
from src.download_data import ensure_raw_data
from src.evaluate import (
    choose_threshold,
    compute_metrics,
    confusion,
    curve_points,
    diagnose_fit,
    permutation_importance_table,
    report_dict,
    threshold_table,
    tree_importance_table,
)
from src.insights import (
    boosting_replay,
    decision_boundaries,
    export_tree,
    falling_paths,
    feature_insights,
    logreg_coefficients,
    pca_projection,
)
from src.models import MODEL_SPECS, base_key, build_model_zoo
from src.train import (
    Dataset,
    ModelResult,
    depth_validation_curve,
    fit_and_profile,
    leakage_ablation,
    learning_curve_data,
    out_of_fold_proba,
    prepare_dataset,
    rank_models,
    run_model_comparison,
    select_final_model,
    stratified_sample,
    temporal_validation,
    tune_model,
)
from src.visualize import generate_figures, model_color

logger = logging.getLogger("main")


def step(title: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that logs a banner and the elapsed time around a pipeline step.

    Args:
        title: Step title shown in the log.

    Returns:
        The decorator.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            logger.info("=" * 78)
            logger.info(title)
            logger.info("-" * 78)
            started = time.perf_counter()
            result = func(*args, **kwargs)
            logger.info("Done in %.1fs", time.perf_counter() - started)
            return result

        return wrapper

    return decorator


def _to_builtin(value: Any) -> Any:  # noqa: ANN401
    """Convert numpy scalars/arrays inside nested structures to JSON-friendly types.

    Args:
        value: Any nested structure.

    Returns:
        The same structure with only built-in types.
    """
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(data: Any, path: Path) -> Path:  # noqa: ANN401
    """Write compact JSON with Unix newlines (byte-identical on Windows and Linux).

    Args:
        data: JSON-serialisable data (numpy types are converted).
        path: Output file.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_builtin(data), separators=(",", ":"), allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    logger.info(
        "Saved %s (%.0f KB)", path.relative_to(config.PROJECT_ROOT), path.stat().st_size / 1e3
    )
    return path


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
@step("1. Data: download (if needed), clean with the Task 2 rules, engineer features, split")
def load_data() -> Dataset:
    """Prepare the dataset.

    Returns:
        The prepared dataset.
    """
    ensure_raw_data()
    return prepare_dataset()


@step("2. Baseline + 5 algorithms: stratified 5-fold cross-validation on the training set")
def compare_models(dataset: Dataset) -> dict[str, ModelResult]:
    """Cross-validate every model in the zoo.

    Args:
        dataset: Prepared dataset.

    Returns:
        Model key -> result.
    """
    zoo = build_model_zoo(dataset.groups)
    return run_model_comparison(zoo, dataset.x_train, dataset.y_train)


@step("3. Hyperparameter tuning (RandomizedSearchCV) on the top models")
def tune_top_models(
    results: dict[str, ModelResult], dataset: Dataset, quick: bool
) -> dict[str, ModelResult]:
    """Tune the best ``config.N_MODELS_TO_TUNE`` models by CV score.

    Args:
        results: Untuned results.
        dataset: Prepared dataset.
        quick: Use a tiny search budget.

    Returns:
        All results including the tuned variants, in display order.
    """
    top = rank_models(results)[: config.N_MODELS_TO_TUNE]
    logger.info("Top %d by CV: %s", len(top), [results[k].name for k in top])
    for key in top:
        n_iter = config.QUICK_TUNING_ITERATIONS if quick else None
        tuned = tune_model(key, dataset.groups, dataset.x_train, dataset.y_train, n_iter=n_iter)
        results[tuned.key] = tuned
    order = [k for base in MODEL_SPECS for k in (base, f"{base}{config.TUNED_SUFFIX}")]
    return {key: results[key] for key in order if key in results}


@step("4. Fit every model on the full training set: fit time, prediction speed, file size")
def fit_all(results: dict[str, ModelResult], dataset: Dataset) -> None:
    """Fit and profile every model (sequentially, so timings are comparable).

    Args:
        results: Model results (updated in place).
        dataset: Prepared dataset.
    """
    for result in results.values():
        fit_and_profile(result, dataset.x_train, dataset.y_train)


@step("5. Over/underfitting diagnosis and model selection (CV evidence only)")
def diagnose_and_select(results: dict[str, ModelResult]) -> dict[str, Any]:
    """Label every model and pick the final one.

    Args:
        results: Fitted and profiled results.

    Returns:
        Selection info plus the per-model diagnosis.
    """
    best_cv = max(r.cv_score for k, r in results.items() if k != config.BASELINE_MODEL)
    for key, result in results.items():
        result.extra["diagnosis"] = diagnose_fit(
            key, result.train[config.PRIMARY_METRIC], result.cv_score, best_cv
        )
        logger.info(
            "  %-30s gap %+.4f -> %s", result.name, result.gap, result.extra["diagnosis"]["label"]
        )
    return select_final_model(results)


@step("6. Threshold tuning on out-of-fold training predictions")
def tune_threshold(final: ModelResult, dataset: Dataset) -> dict[str, Any]:
    """Choose the decision threshold without touching the test set.

    Args:
        final: Final model result.
        dataset: Prepared dataset.

    Returns:
        Threshold info and the out-of-fold table.
    """
    oof = out_of_fold_proba(final.pipeline, dataset.x_train, dataset.y_train)
    table = threshold_table(dataset.y_train, oof)
    chosen = choose_threshold(table, config.THRESHOLD_STRATEGY)
    recall_threshold = choose_threshold(table, "recall_target", config.RECALL_TARGET)
    at = table.set_index("threshold")
    logger.info(
        "Chosen threshold %.2f (%s): OOF precision %.3f, recall %.3f, F1 %.3f (at 0.50: F1 %.3f)",
        chosen,
        config.THRESHOLD_STRATEGY,
        at.loc[chosen, "precision"],
        at.loc[chosen, "recall"],
        at.loc[chosen, "f1"],
        at.loc[config.DEFAULT_THRESHOLD, "f1"],
    )
    return {
        "chosen": chosen,
        "strategy": config.THRESHOLD_STRATEGY,
        "recall_target": config.RECALL_TARGET,
        "recall_target_threshold": recall_threshold,
        "oof": table.to_dict("records"),
        "oof_metrics_default": compute_metrics(dataset.y_train, oof, config.DEFAULT_THRESHOLD),
        "oof_metrics_chosen": compute_metrics(dataset.y_train, oof, chosen),
    }


@step("7. Learning curves and the Decision Tree depth validation curve")
def overfitting_curves(
    results: dict[str, ModelResult], dataset: Dataset, quick: bool
) -> dict[str, Any]:
    """Learning curves (tuned variant when available) and the depth validation curve.

    Args:
        results: Model results.
        dataset: Prepared dataset.
        quick: Use fewer curve points.

    Returns:
        ``{"learning_curves": {key: curve}, "validation_curve": curve}``.
    """
    curves = {}
    for base in config.LEARNING_CURVE_MODELS:
        key = f"{base}{config.TUNED_SUFFIX}" if f"{base}{config.TUNED_SUFFIX}" in results else base
        logger.info("  learning curve: %s", results[key].name)
        sizes = config.QUICK_CURVE_SIZES if quick else config.LEARNING_CURVE_SIZES
        curves[key] = learning_curve_data(
            results[key].pipeline, dataset.x_train, dataset.y_train, sizes
        )
    validation = depth_validation_curve(dataset.groups, dataset.x_train, dataset.y_train)
    logger.info("  validation curve: best Decision Tree depth = %d", validation["best_depth"])
    return {"learning_curves": curves, "validation_curve": validation}


@step("8. FINAL EVALUATION - the held-out test set is used here, once, for every model")
def evaluate_on_test(
    results: dict[str, ModelResult], final_key: str, threshold: float, dataset: Dataset
) -> dict[str, Any]:
    """Score every frozen model on the test set and build the final-model report.

    Args:
        results: Fitted results (updated in place with test outputs).
        final_key: Selected model key.
        threshold: Tuned decision threshold.
        dataset: Prepared dataset.

    Returns:
        Final-model evaluation details.
    """
    y_test = dataset.y_test
    for result in results.values():
        proba = result.fitted.predict_proba(dataset.x_test)[:, 1]
        result.extra.update(
            test=compute_metrics(y_test, proba),
            confusion=confusion(y_test, proba),
            curves=curve_points(y_test, proba),
            test_proba=proba,
        )
        logger.info(
            "  %-30s test %s %.4f | ROC-AUC %.4f | F1 %.4f",
            result.name,
            config.METRIC_LABELS[config.PRIMARY_METRIC],
            result.extra["test"][config.PRIMARY_METRIC],
            result.extra["test"]["roc_auc"],
            result.extra["test"]["f1"],
        )
    final = results[final_key]
    proba = final.extra["test_proba"]
    tuned_metrics = compute_metrics(y_test, proba, threshold)
    logger.info(
        "Final model %s @ %.2f: precision %.3f, recall %.3f, F1 %.3f, PR-AUC %.3f",
        final.name,
        threshold,
        tuned_metrics["precision"],
        tuned_metrics["recall"],
        tuned_metrics["f1"],
        tuned_metrics["average_precision"],
    )
    return {
        "key": final_key,
        "name": final.name,
        "threshold": threshold,
        "test_default": final.extra["test"],
        "test_tuned": tuned_metrics,
        "confusion_default": final.extra["confusion"],
        "confusion_tuned": confusion(y_test, proba, threshold),
        "report_tuned": report_dict(y_test, proba, threshold),
        "test_threshold_table": threshold_table(y_test, proba).to_dict("records"),
        "params": final.params,
        "size_mb": final.size_mb,
    }


@step("9. Explainability: permutation importance (test) and tree importances")
def explain(results: dict[str, ModelResult], final_key: str, dataset: Dataset) -> dict[str, Any]:
    """Permutation importance of the final model and impurity importances of the trees.

    Args:
        results: Fitted results.
        final_key: Selected model key.
        dataset: Prepared dataset.

    Returns:
        Importance tables, insights and Logistic Regression weights.
    """
    final = results[final_key]
    permutation = permutation_importance_table(final.fitted, dataset.x_test, dataset.y_test)
    logger.info("Permutation importance (top 8):\n%s", permutation.head(8).to_string(index=False))
    tree_tables = {}
    for base in ("tree", "rf"):
        key = f"{base}{config.TUNED_SUFFIX}" if f"{base}{config.TUNED_SUFFIX}" in results else base
        tree_tables[key] = tree_importance_table(results[key].fitted, dataset.groups).to_dict(
            "records"
        )
    top_features = permutation["feature"].head(8).tolist()
    return {
        "permutation": permutation.to_dict("records"),
        "tree_importance": tree_tables,
        "insights": feature_insights(dataset.train_rows, top_features, dataset.groups),
        "logreg_coefficients": logreg_coefficients(results["logreg"].fitted),
    }


@step("10. Extra checks: temporal validation (data drift) and leakage ablation")
def extra_checks(
    results: dict[str, ModelResult], final_key: str, dataset: Dataset
) -> dict[str, Any]:
    """Run the temporal split and the post-booking-column ablation.

    Args:
        results: Model results.
        final_key: Selected model key.
        dataset: Prepared dataset.

    Returns:
        ``{"temporal": ..., "ablation": ...}``.
    """
    temporal = temporal_validation(results, dataset.featured, dataset.groups)
    final = results[final_key]
    ablation = leakage_ablation(
        base_key(final_key),
        final.params,
        dataset.groups,
        dataset.train_rows,
    )
    return {"temporal": temporal, "ablation": ablation}


@step("11. Interactive exports: tree, falling bookings, playground, 3-D PCA, boosting replay")
def interactive_exports(results: dict[str, ModelResult], dataset: Dataset) -> dict[str, Any]:
    """Precompute every interactive view of the web app.

    Args:
        results: Fitted results.
        dataset: Prepared dataset.

    Returns:
        Explore payload for ``app/cache/explore.json``.
    """
    tree_model = results["tree"].fitted
    x_fall, y_fall = stratified_sample(dataset.x_test, dataset.y_test, config.FALLING_SAMPLES)
    hgb_key = "hgb_tuned" if "hgb_tuned" in results else "hgb"
    replay_params = {k: v for k, v in results[hgb_key].params.items() if k != "random_state"}
    return {
        "tree": export_tree(tree_model, dataset.groups, dataset.x_train, dataset.y_train),
        "falling": falling_paths(
            tree_model, dataset.groups, dataset.x_train, dataset.y_train, x_fall, y_fall
        ),
        "playground": decision_boundaries(dataset.train_rows, dataset.test_rows),
        "pca": pca_projection(dataset.x_train, dataset.y_train, dataset.groups),
        "replay": {
            "model": results[hgb_key].name,
            **boosting_replay(replay_params, dataset.groups, dataset.x_train, dataset.y_train),
        },
    }


# ---------------------------------------------------------------------------
# Summary and outputs
# ---------------------------------------------------------------------------
def model_record(result: ModelResult, all_keys: list[str], rank: int | None) -> dict[str, Any]:
    """One model's complete, JSON-ready record.

    Args:
        result: Evaluated model result.
        all_keys: All model keys (for colours).
        rank: Leaderboard rank by CV score (``None`` for the baseline).

    Returns:
        The record.
    """
    spec = MODEL_SPECS[base_key(result.key)]
    test = result.extra["test"]
    return {
        "key": result.key,
        "name": result.name,
        "base": spec.key,
        "tuned": result.key.endswith(config.TUNED_SUFFIX),
        "rank": rank,
        "color": model_color(result.key, all_keys),
        "color_dark": model_color(result.key, all_keys, dark=True),
        "preprocessing": spec.preprocessing,
        "interpretability": spec.interpretability,
        "training_rows": result.training_rows,
        "params": {k: v for k, v in result.params.items() if k != "random_state"},
        "cv": result.cv,
        "train": result.train,
        "test": test,
        "gap": result.gap,
        "cv_test_diff": round(result.cv_score - test[config.PRIMARY_METRIC], 4),
        "diagnosis": result.extra["diagnosis"],
        "cv_fit_time_s": result.cv_fit_time_s,
        "fit_time_s": result.fit_time_s,
        "predict_ms_per_1k": result.predict_ms_per_1k,
        "size_mb": result.size_mb,
        "confusion": result.extra["confusion"],
        "curves": result.extra["curves"],
        "search": result.search,
    }


def build_summary(
    dataset: Dataset,
    results: dict[str, ModelResult],
    selection: dict[str, Any],
    parts: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the run summary used by the figures, metrics.json and the web app.

    Args:
        dataset: Prepared dataset.
        results: Evaluated results.
        selection: Output of ``select_final_model``.
        parts: Threshold, curves, final, explain and extra-check payloads.

    Returns:
        The summary dictionary.
    """
    ranking = selection["ranking"]
    all_keys = list(results)
    models = [
        model_record(r, all_keys, ranking.index(k) + 1 if k in ranking else None)
        for k, r in results.items()
    ]
    return {
        "dataset": {
            "raw_rows": dataset.raw_shape[0],
            "raw_columns": dataset.raw_shape[1],
            "clean_rows": len(dataset.featured),
            "train_rows": len(dataset.train_rows),
            "test_rows": len(dataset.test_rows),
            "train_cancel_rate": round(float(dataset.y_train.mean()), 4),
            "test_cancel_rate": round(float(dataset.y_test.mean()), 4),
            "inputs": len(dataset.groups.all),
            "numeric": dataset.groups.numeric,
            "binary": dataset.groups.binary,
            "categorical": dataset.groups.categorical,
            "dropped": config.DROPPED_FEATURES,
            "leakage_columns": config.LEAKAGE_COLUMNS,
            "adr_cap": dataset.adr_cap,
            "knn_sample": config.KNN_SAMPLE_SIZE,
        },
        "setup": {
            "primary_metric": config.PRIMARY_METRIC,
            "primary_label": config.METRIC_LABELS[config.PRIMARY_METRIC],
            "metrics": config.METRICS,
            "metric_labels": config.METRIC_LABELS,
            "cv_folds": config.CV_FOLDS,
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "overfit_gap": config.OVERFIT_GAP,
            "underfit_margin": config.UNDERFIT_MARGIN,
            "business_defaults": {
                "empty_room_cost": config.EMPTY_ROOM_COST,
                "overbooking_cost": config.OVERBOOKING_COST,
            },
        },
        "models": models,
        "selection": {k: v for k, v in selection.items()},
        **parts,
    }


def slim_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """The headline numbers for ``models/metrics.json`` (no curves or tables).

    Args:
        summary: Full run summary.

    Returns:
        A small dictionary.
    """
    heavy = {"curves", "search", "confusion"}
    return {
        "dataset": summary["dataset"],
        "setup": summary["setup"],
        "selection": summary["selection"],
        "final": {k: v for k, v in summary["final"].items() if k != "test_threshold_table"},
        "threshold": {k: v for k, v in summary["threshold"].items() if k != "oof"},
        "models": [{k: v for k, v in m.items() if k not in heavy} for m in summary["models"]],
        "temporal": summary["temporal"],
        "ablation": summary["ablation"],
        "validation_curve_best_depth": summary["validation_curve"]["best_depth"],
        "permutation_top10": summary["permutation"][:10],
    }


def comparison_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Flat model-comparison table (CSV).

    Args:
        summary: Run summary.

    Returns:
        One row per model.
    """
    rows = []
    for m in summary["models"]:
        row: dict[str, Any] = {"model": m["name"], "rank": m["rank"]}
        for metric in config.METRICS:
            row[f"cv_{metric}_mean"] = m["cv"][metric]["mean"]
            row[f"cv_{metric}_std"] = m["cv"][metric]["std"]
        row |= {f"train_{metric}": m["train"][metric] for metric in config.METRICS}
        row |= {f"test_{metric}": m["test"][metric] for metric in config.METRICS}
        row |= {
            "train_cv_gap": m["gap"],
            "diagnosis": m["diagnosis"]["label"],
            "cv_fold_fit_time_s": m["cv_fit_time_s"],
            "full_fit_time_s": m["fit_time_s"],
            "predict_ms_per_1k": m["predict_ms_per_1k"],
            "size_mb": m["size_mb"],
            "training_rows": m["training_rows"],
            "temporal_test_" + config.PRIMARY_METRIC: summary["temporal"]["metrics"]
            .get(m["key"], {})
            .get(config.PRIMARY_METRIC),
        }
        rows.append(row)
    return pd.DataFrame(rows)


@step("12. Save model, metrics, comparison table, figures and web-app cache")
def save_outputs(
    summary: dict[str, Any], explore: dict[str, Any], final_model: Pipeline, tree_model: Pipeline
) -> None:
    """Write every artifact.

    Args:
        summary: Run summary.
        explore: Interactive exports.
        final_model: Fitted final pipeline.
        tree_model: Fitted Decision Tree pipeline (static tree figure).
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, config.FINAL_MODEL_PATH, compress=3)
    logger.info(
        "Saved %s (%.2f MB)",
        config.FINAL_MODEL_PATH.relative_to(config.PROJECT_ROOT),
        config.FINAL_MODEL_PATH.stat().st_size / 1e6,
    )
    write_json(slim_metrics(summary), config.METRICS_PATH)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_table(summary).to_csv(config.MODEL_COMPARISON_PATH, index=False, lineterminator="\n")
    logger.info("Saved %s", config.MODEL_COMPARISON_PATH.relative_to(config.PROJECT_ROOT))
    write_json(summary, config.APP_DASHBOARD_CACHE)
    write_json(explore, config.APP_EXPLORE_CACHE)
    build_predict_cache()
    generate_figures(summary, tree_model)


def regenerate_figures() -> None:
    """Rebuild the figures from the saved summary (no retraining except the quick tree).

    The Decision Tree is refitted with the same seed for the static tree plot (~2 s).
    """
    config.setup_logging()
    summary = json.loads(config.APP_DASHBOARD_CACHE.read_text(encoding="utf-8"))
    dataset = prepare_dataset()
    tree = build_model_zoo(dataset.groups)["tree"].fit(dataset.x_train, dataset.y_train)
    generate_figures(summary, tree)


def log_final_table(summary: dict[str, Any]) -> None:
    """Print the model comparison and the selected model at the end of the run.

    Args:
        summary: Run summary.
    """
    table = comparison_table(summary)
    metric = config.PRIMARY_METRIC
    columns = [
        "model",
        f"cv_{metric}_mean",
        f"train_{metric}",
        f"test_{metric}",
        "test_roc_auc",
        "test_f1",
        "train_cv_gap",
        "diagnosis",
        "full_fit_time_s",
        "size_mb",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        logger.info("Model comparison:\n%s", table[columns].to_string(index=False))
    final = summary["final"]
    logger.info(
        "SELECTED: %s | threshold %.2f | test PR-AUC %.4f, ROC-AUC %.4f, F1 %.4f, recall %.4f",
        final["name"],
        final["threshold"],
        final["test_tuned"]["average_precision"],
        final["test_tuned"]["roc_auc"],
        final["test_tuned"]["f1"],
        final["test_tuned"]["recall"],
    )


def main(quick: bool = False) -> None:
    """Run every step in order.

    Args:
        quick: Smoke-test mode (tiny tuning budget, fewer curve points).
    """
    config.setup_logging()
    started = time.perf_counter()
    dataset = load_data()
    results = compare_models(dataset)
    results = tune_top_models(results, dataset, quick)
    fit_all(results, dataset)
    selection = diagnose_and_select(results)
    final = results[selection["key"]]
    threshold = tune_threshold(final, dataset)
    curves = overfitting_curves(results, dataset, quick)
    final_eval = evaluate_on_test(results, selection["key"], threshold["chosen"], dataset)
    explanations = explain(results, selection["key"], dataset)
    checks = extra_checks(results, selection["key"], dataset)
    explore = interactive_exports(results, dataset)
    parts = {"threshold": threshold, "final": final_eval, **curves, **explanations, **checks}
    summary = build_summary(dataset, results, selection, parts)
    summary["replay"] = explore["replay"]
    save_outputs(summary, explore, final.fitted, results["tree"].fitted)
    log_final_table(summary)
    logger.info("All done in %.1f minutes", (time.perf_counter() - started) / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test with a tiny budget")
    parser.add_argument("--figures-only", action="store_true", help="redraw figures from cache")
    arguments = parser.parse_args()
    if arguments.figures_only:
        regenerate_figures()
    else:
        main(quick=arguments.quick)
