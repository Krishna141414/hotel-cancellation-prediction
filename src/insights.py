"""Precomputed exports behind the interactive explainability views of the web app.

Decision-tree structure, "falling bookings" paths, 2-feature decision boundaries,
a 3-D PCA projection, a gradient-boosting training replay and plain-English
feature insights. All of it is computed once by ``main.py`` and cached as JSON,
so the web server never trains anything.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.features import FeatureGroups, build_scaled_preprocessor, readable_feature
from src.models import build_pipeline
from src.train import stratified_sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision tree structure
# ---------------------------------------------------------------------------
def _is_integer_column(x: pd.DataFrame, column: str) -> bool:
    """True when every value of a numeric column is a whole number.

    Args:
        x: Input features.
        column: Column name.

    Returns:
        Whether thresholds can be shown as integers.
    """
    values = x[column]
    return bool(pd.api.types.is_numeric_dtype(values) and (values % 1 == 0).all())


def split_question(
    feature_name: str, threshold: float, groups: FeatureGroups, x: pd.DataFrame
) -> dict[str, str]:
    """Turn one tree split into a readable question with yes/no branch labels.

    sklearn sends ``value <= threshold`` to the left child.

    Args:
        feature_name: Transformed column name used by the split.
        threshold: Split threshold.
        groups: Model input columns.
        x: Raw training inputs (to format numeric thresholds).

    Returns:
        ``{"question", "left", "right"}`` texts.
    """
    column, category = readable_feature(feature_name, groups)
    if category is not None:
        return {"question": f"{column} = {category}?", "left": "no", "right": "yes"}
    if column in groups.binary:
        return {"question": f"{column} = 1?", "left": "no", "right": "yes"}
    if _is_integer_column(x, column):
        shown = f"{int(np.floor(threshold))}"
    else:
        shown = f"{threshold:.1f}"
    return {"question": f"{column} <= {shown}?", "left": "yes", "right": "no"}


def _node_stats(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Number of training rows and actual cancellations passing through each node.

    Args:
        model: Fitted Decision Tree pipeline.
        x: Training inputs.
        y: Training labels.

    Returns:
        ``(rows_per_node, cancellations_per_node)``.
    """
    indicator = model.named_steps["clf"].decision_path(model.named_steps["prep"].transform(x))
    rows = np.asarray(indicator.sum(axis=0)).ravel()
    canceled = np.asarray(indicator.T @ np.asarray(y)).ravel()
    return rows, canceled


def export_tree(
    model: Pipeline,
    groups: FeatureGroups,
    x: pd.DataFrame,
    y: pd.Series,
    max_depth: int = config.TREE_EXPORT_DEPTH,
) -> dict[str, Any]:
    """Export the top levels of a fitted Decision Tree as nested JSON (for D3).

    Args:
        model: Fitted Decision Tree pipeline.
        groups: Model input columns.
        x: Training inputs (node statistics are actual training counts).
        y: Training labels.
        max_depth: Deepest level to export.

    Returns:
        Root node; each node has ``id, rows, cancel_rate, predicted`` and, for splits,
        ``question`` and ``children`` (each child carries its ``branch`` label).
    """
    tree = model.named_steps["clf"].tree_
    names = model.named_steps["prep"].get_feature_names_out()
    rows, canceled = _node_stats(model, x, y)

    def build(node: int, depth: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": int(node),
            "rows": int(rows[node]),
            "cancel_rate": round(float(canceled[node] / max(rows[node], 1)), 4),
            "predicted": int(np.argmax(tree.value[node][0])),
        }
        is_split = tree.children_left[node] != tree.children_right[node]
        if not is_split:
            return result
        if depth >= max_depth:
            result["truncated"] = True
            return result
        texts = split_question(names[tree.feature[node]], tree.threshold[node], groups, x)
        result["question"] = texts["question"]
        left = build(tree.children_left[node], depth + 1)
        right = build(tree.children_right[node], depth + 1)
        left["branch"], right["branch"] = texts["left"], texts["right"]
        result["children"] = [left, right]
        return result

    return build(0, 0)


def falling_paths(
    model: Pipeline,
    groups: FeatureGroups,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_sample: pd.DataFrame,
    y_sample: pd.Series,
    depth: int = config.FALLING_TREE_DEPTH,
) -> dict[str, Any]:
    """Paths of sample bookings through the top ``depth`` levels of the tree.

    The truncated tree predicts with each depth-``depth`` node's majority class, so
    the animation shows how well the top splits alone separate the classes.

    Args:
        model: Fitted Decision Tree pipeline.
        groups: Model input columns.
        x_train: Training inputs (for node statistics).
        y_train: Training labels.
        x_sample: Bookings to animate.
        y_sample: Their true labels.
        depth: Levels to show.

    Returns:
        ``{"tree": nested nodes, "samples": [{"path", "label", "predicted"}], "accuracy"}``.
    """
    tree_json = export_tree(model, groups, x_train, y_train, max_depth=depth)
    predicted_by_node: dict[int, int] = {}

    def collect(node: dict[str, Any]) -> None:
        predicted_by_node[node["id"]] = node["predicted"]
        for child in node.get("children", []):
            collect(child)

    collect(tree_json)
    indicator = model.named_steps["clf"].decision_path(
        model.named_steps["prep"].transform(x_sample)
    )
    samples = []
    for row, label in zip(range(indicator.shape[0]), np.asarray(y_sample), strict=True):
        nodes = indicator.indices[indicator.indptr[row] : indicator.indptr[row + 1]]
        path = [int(n) for n in sorted(nodes) if int(n) in predicted_by_node][: depth + 1]
        # Node ids grow with depth along a path, so sorting keeps root -> leaf order.
        samples.append(
            {"path": path, "label": int(label), "predicted": predicted_by_node[path[-1]]}
        )
    accuracy = float(np.mean([s["label"] == s["predicted"] for s in samples]))
    return {"tree": tree_json, "samples": samples, "accuracy": round(accuracy, 4)}


# ---------------------------------------------------------------------------
# Decision-boundary playground (2 features at a time)
# ---------------------------------------------------------------------------
PLAYGROUND_MODELS: dict[str, str] = {
    "logreg": "Logistic Regression (straight line)",
    "tree_shallow": "Decision Tree, depth 3",
    "tree_deep": "Decision Tree, no depth limit",
    "knn": "KNN, k = 15",
    "hgb": "HistGradientBoosting",
}


def _playground_estimator(key: str) -> Pipeline:
    """Small 2-feature model for the decision-boundary playground.

    Args:
        key: Key from :data:`PLAYGROUND_MODELS`.

    Returns:
        Unfitted estimator (scaled where the algorithm needs it).
    """
    seed = config.RANDOM_STATE
    estimators = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")),
        "tree_shallow": make_pipeline(
            DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)
        ),
        "tree_deep": make_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=seed)
        ),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "hgb": make_pipeline(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
        ),
    }
    return estimators[key]


def decision_boundaries(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> dict[str, Any]:
    """Probability grids of small 2-feature models, for each configured feature pair.

    Args:
        train_rows: Training bookings (a stratified sample is used to fit).
        test_rows: Test bookings (a sample is drawn as dots and scored).

    Returns:
        ``{"models": {...}, "pairs": [{"x", "y", "x_range", "y_range", "grid", "points",
        "results": {model: {"grid", "train", "test"}}}]}``; grids hold 0-100 integers.
    """
    n = config.PLAYGROUND_GRID
    low, high = config.PLAYGROUND_QUANTILES
    pairs = []
    for x_name, y_name in config.PLAYGROUND_PAIRS:
        columns = [x_name, y_name]
        x_fit, y_fit = stratified_sample(
            train_rows[columns], train_rows[config.TARGET], config.PLAYGROUND_SAMPLE
        )
        x_eval, y_eval = stratified_sample(
            test_rows[columns], test_rows[config.TARGET], config.PLAYGROUND_SAMPLE
        )
        x_range = [float(x_fit[x_name].quantile(low)), float(x_fit[x_name].quantile(high))]
        y_range = [float(x_fit[y_name].quantile(low)), float(x_fit[y_name].quantile(high))]
        gx, gy = np.meshgrid(np.linspace(*x_range, n), np.linspace(*y_range, n))
        grid = pd.DataFrame({x_name: gx.ravel(), y_name: gy.ravel()})
        results = {}
        for key in PLAYGROUND_MODELS:
            model = _playground_estimator(key).fit(x_fit, y_fit)
            results[key] = {
                "grid": (model.predict_proba(grid)[:, 1] * 100).round().astype(int).tolist(),
                "train": round(
                    float(average_precision_score(y_fit, model.predict_proba(x_fit)[:, 1])), 4
                ),
                "test": round(
                    float(average_precision_score(y_eval, model.predict_proba(x_eval)[:, 1])), 4
                ),
            }
        points = x_eval.assign(label=y_eval.to_numpy()).sample(
            config.PLAYGROUND_POINTS, random_state=config.RANDOM_STATE
        )
        pairs.append(
            {
                "x": x_name,
                "y": y_name,
                "x_range": [round(v, 2) for v in x_range],
                "y_range": [round(v, 2) for v in y_range],
                "size": n,
                "points": points.round(2).to_numpy().tolist(),
                "results": results,
            }
        )
        logger.info(
            "  Playground %s x %s: %s",
            x_name,
            y_name,
            {k: (v["train"], v["test"]) for k, v in results.items()},
        )
    return {"models": PLAYGROUND_MODELS, "pairs": pairs}


# ---------------------------------------------------------------------------
# 3-D booking space and boosting replay
# ---------------------------------------------------------------------------
def pca_projection(
    x_train: pd.DataFrame, y_train: pd.Series, groups: FeatureGroups
) -> dict[str, Any]:
    """Project scaled, one-hot encoded bookings onto 3 principal components.

    Args:
        x_train: Training inputs (preprocessor and PCA are fitted on these).
        y_train: Training labels.
        groups: Model input columns.

    Returns:
        ``{"points": [[pc1, pc2, pc3, label], ...], "explained", "loadings"}``.
    """
    preprocessor = build_scaled_preprocessor(groups)
    transformed = preprocessor.fit_transform(x_train)
    pca = PCA(n_components=3, random_state=config.RANDOM_STATE).fit(transformed)
    x_sample, y_sample = stratified_sample(x_train, y_train, config.PCA_SAMPLES)
    projected = pca.transform(preprocessor.transform(x_sample))
    names = preprocessor.get_feature_names_out()
    loadings = []
    for component in pca.components_:
        top = np.argsort(np.abs(component))[::-1][:3]
        loadings.append([names[i].split("__", 1)[-1] for i in top])
    points = np.column_stack([projected.round(3), y_sample.to_numpy()])
    return {
        "points": points.tolist(),
        "explained": pca.explained_variance_ratio_.round(4).tolist(),
        "loadings": loadings,
    }


def boosting_replay(
    params: dict[str, Any], groups: FeatureGroups, x: pd.DataFrame, y: pd.Series
) -> dict[str, Any]:
    """Train and validation score of HistGradientBoosting after every boosting stage.

    The training set is split once more (stratified) into a fit part and a
    validation part, so this never touches the test set.

    Args:
        params: HistGradientBoosting hyperparameters (the tuned ones if available).
        groups: Model input columns.
        x: Training features.
        y: Training labels.

    Returns:
        ``{"iterations", "train", "validation", "best_iteration", "best_score"}``.
    """
    x_fit, x_val, y_fit, y_val = train_test_split(
        x,
        y,
        test_size=config.REPLAY_VALIDATION_SHARE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    model = build_pipeline("hgb", groups, params).fit(x_fit, y_fit)
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    fit_sample, y_fit_sample = stratified_sample(x_fit, y_fit, len(x_val))
    staged_train = clf.staged_predict_proba(prep.transform(fit_sample))
    staged_val = clf.staged_predict_proba(prep.transform(x_val))
    train_scores, val_scores = [], []
    for train_proba, val_proba in zip(staged_train, staged_val, strict=True):
        train_scores.append(average_precision_score(y_fit_sample, train_proba[:, 1]))
        val_scores.append(average_precision_score(y_val, val_proba[:, 1]))
    iterations = np.arange(1, len(val_scores) + 1)
    keep = np.unique(
        np.linspace(0, len(iterations) - 1, config.REPLAY_MAX_POINTS).round().astype(int)
    )
    best = int(np.argmax(val_scores))
    return {
        "iterations": iterations[keep].tolist(),
        "train": np.round(np.asarray(train_scores)[keep], 4).tolist(),
        "validation": np.round(np.asarray(val_scores)[keep], 4).tolist(),
        "best_iteration": int(iterations[best]),
        "best_score": round(float(val_scores[best]), 4),
        "final_score": round(float(val_scores[-1]), 4),
    }


# ---------------------------------------------------------------------------
# Plain-English insights
# ---------------------------------------------------------------------------
def feature_insight(rows: pd.DataFrame, column: str, groups: FeatureGroups) -> dict[str, Any]:
    """Describe how the cancellation rate changes with one input column.

    Args:
        rows: Training bookings with the target.
        column: Input column.
        groups: Model input columns (to know the column type).

    Returns:
        ``{"feature", "text", "high_rate", "low_rate"}`` (rates as fractions).
    """
    target = rows[config.TARGET]
    if column in groups.categorical:
        shares = rows[column].value_counts(normalize=True)
        common = shares[shares >= config.MIN_CATEGORY_FREQUENCY].index
        rates = target[rows[column].isin(common)].groupby(rows[column]).mean().sort_values()
        high, low = rates.index[-1], rates.index[0]
        text = (
            f"{column} = {high} cancels {rates.iloc[-1]:.0%} of the time, "
            f"vs {rates.iloc[0]:.0%} for {low}."
        )
        return {
            "feature": column,
            "text": text,
            "high_rate": rates.iloc[-1],
            "low_rate": rates.iloc[0],
        }
    if column in groups.binary or rows[column].nunique() <= 2:
        yes, no = target[rows[column] > 0].mean(), target[rows[column] == 0].mean()
        text = f"Bookings with {column} = 1 cancel {yes:.0%}, vs {no:.0%} when it is 0."
        return {"feature": column, "text": text, "high_rate": yes, "low_rate": no}
    if rows[column].nunique() <= config.INSIGHT_MAX_COUNT_VALUES:
        # Small counts (requests, parking spaces, ...): "none" vs "1 or more".
        some, none = target[rows[column] > 0].mean(), target[rows[column] == 0].mean()
        text = f"Bookings with 1+ {column} cancel {some:.0%}, vs {none:.0%} with none."
        return {"feature": column, "text": text, "high_rate": some, "low_rate": none}
    low_cut, high_cut = rows[column].quantile([0.25, 0.75])
    if low_cut == high_cut:
        yes, no = target[rows[column] > low_cut].mean(), target[rows[column] <= low_cut].mean()
        text = f"Bookings with {column} > {low_cut:g} cancel {yes:.0%}, vs {no:.0%} otherwise."
        return {"feature": column, "text": text, "high_rate": yes, "low_rate": no}
    high_rate = target[rows[column] >= high_cut].mean()
    low_rate = target[rows[column] <= low_cut].mean()
    text = (
        f"Top quarter of {column} (>= {high_cut:g}) cancels {high_rate:.0%}, "
        f"bottom quarter (<= {low_cut:g}) {low_rate:.0%}."
    )
    return {"feature": column, "text": text, "high_rate": high_rate, "low_rate": low_rate}


def feature_insights(
    rows: pd.DataFrame, columns: list[str], groups: FeatureGroups
) -> list[dict[str, Any]]:
    """Insights for several columns (rates rounded for JSON).

    Args:
        rows: Training bookings with the target.
        columns: Columns to describe, most important first.
        groups: Model input columns.

    Returns:
        One insight per column.
    """
    insights = []
    for column in columns:
        item = feature_insight(rows, column, groups)
        item["high_rate"] = round(float(item["high_rate"]), 4)
        item["low_rate"] = round(float(item["low_rate"]), 4)
        insights.append(item)
    return insights


def logreg_coefficients(model: Pipeline, top: int = config.TOP_FEATURES_TO_SHOW) -> list[dict]:
    """Largest positive and negative Logistic Regression weights (scaled inputs).

    Args:
        model: Fitted Logistic Regression pipeline.
        top: Number of weights to return (split between signs by magnitude).

    Returns:
        ``[{"feature", "coef"}]`` sorted by absolute weight.
    """
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:top]
    return [
        {"feature": names[i].split("__", 1)[-1], "coef": round(float(coefs[i]), 4)} for i in order
    ]
