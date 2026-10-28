"""Shared plot style and the model-evaluation figures saved to ``reports/figures/``.

Same approach as Task 2 (headline title that states the finding, muted subtitle,
hairline grid, text in neutral ink), now in the project palette that the web app
also uses: electric blue, magenta-pink (canceled), mint teal (not canceled) and
marigold (selected model). Every function takes the JSON-ready summary built by
``main.py``, so figures and web app always show the same numbers.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

from src import config
from src.models import base_key

logger = logging.getLogger(__name__)

PALETTE = config.PALETTE
METRIC_LABELS = config.METRIC_LABELS
PRIMARY_LABEL = METRIC_LABELS[config.PRIMARY_METRIC]


# ---------------------------------------------------------------------------
# Colours and style
# ---------------------------------------------------------------------------
def tint(color: str, amount: float) -> str:
    """Mix a colour with white.

    Args:
        color: Hex colour.
        amount: 0 = unchanged, 1 = white.

    Returns:
        Hex colour.
    """
    rgb = np.asarray(to_rgb(color))
    return to_hex(rgb + (1 - rgb) * amount)


def model_color(key: str, all_keys: list[str], dark: bool = False) -> str:
    """Colour of a model; an untuned model whose tuned version exists gets a lighter tint.

    Args:
        key: Model key.
        all_keys: Every model key in the comparison.
        dark: Use the dark-theme colours (web app).

    Returns:
        Hex colour.
    """
    colors = config.MODEL_COLORS_DARK if dark else config.MODEL_COLORS
    color = colors[base_key(key)]
    if not key.endswith(config.TUNED_SUFFIX) and f"{key}{config.TUNED_SUFFIX}" in all_keys:
        return tint(color, config.UNTUNED_TINT)
    return color


def apply_style() -> None:
    """Set matplotlib defaults for every project figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface_light"],
            "axes.facecolor": PALETTE["surface_light"],
            "savefig.facecolor": PALETTE["surface_light"],
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink_secondary"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": PALETTE["ink_muted"],
            "ytick.color": PALETTE["ink_muted"],
            "xtick.labelcolor": PALETTE["ink_secondary"],
            "ytick.labelcolor": PALETTE["ink_secondary"],
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
        }
    )


def new_figure(
    title: str,
    subtitle: str,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (9, 4.8),
) -> tuple[Figure, list[Axes]]:
    """Create a figure with a left-aligned headline, muted subtitle and a gradient rule.

    Args:
        title: Headline stating the insight.
        subtitle: What is plotted (measure, population).
        nrows: Panel rows.
        ncols: Panel columns.
        figsize: Figure size in inches.

    Returns:
        The figure and a flat list of its axes.
    """
    apply_style()
    # Object-oriented Figure (no pyplot): no GUI backend or global state needed.
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)  # a real canvas: plot_tree needs a renderer to measure text
    axes = fig.subplots(nrows, ncols, squeeze=False)
    fig.text(0.01, 0.985, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.01, 0.925, subtitle, fontsize=10, color=PALETTE["ink_secondary"], va="top")
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    rule = fig.add_axes((0.01, 0.885, 0.18, 0.006))
    rule.imshow(gradient, aspect="auto", cmap=_blue_pink_cmap())
    rule.set_axis_off()
    fig.subplots_adjust(top=0.82, left=0.1, right=0.97, bottom=0.12, wspace=0.3, hspace=0.45)
    return fig, list(axes.ravel())


def _blue_pink_cmap() -> LinearSegmentedColormap:
    """The hero gradient (blue -> pink) as a colormap.

    Returns:
        Colormap.
    """
    return LinearSegmentedColormap.from_list("hero", [PALETTE["primary"], PALETTE["pink"]])


def save_figure(fig: Figure, name: str, folder: Path = config.FIGURES_DIR) -> Path:
    """Save a figure as PNG.

    Args:
        fig: Figure to save.
        name: File name without extension.
        folder: Output folder.

    Returns:
        Path of the saved PNG.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    logger.info("Saved figure %s", path.name)
    return path


def _label_bars(ax: Axes, bars: Any, fmt: str = "{:.3f}", horizontal: bool = False) -> None:  # noqa: ANN401
    """Write each bar's value at its tip.

    Args:
        ax: Target axes.
        bars: Bar container.
        fmt: Value format.
        horizontal: Bars are horizontal.
    """
    for bar in bars:
        if horizontal:
            value = bar.get_width()
            xy, offset, ha, va = (
                (value, bar.get_y() + bar.get_height() / 2),
                (4, 0),
                "left",
                "center",
            )
        else:
            value = bar.get_height()
            xy, offset, ha, va = (
                (bar.get_x() + bar.get_width() / 2, value),
                (0, 3),
                "center",
                "bottom",
            )
        ax.annotate(
            fmt.format(value),
            xy,
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            color=PALETTE["ink_secondary"],
        )


def _models(summary: dict[str, Any], include_baseline: bool = True) -> list[dict[str, Any]]:
    """Model records in display order.

    Args:
        summary: Run summary.
        include_baseline: Keep the dummy baseline.

    Returns:
        Model records.
    """
    return [m for m in summary["models"] if include_baseline or m["key"] != config.BASELINE_MODEL]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_class_balance(summary: dict[str, Any]) -> Figure:
    """Share of canceled bookings in the train and test sets.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    data = summary["dataset"]
    fig, (ax,) = new_figure(
        "About 1 in 4 bookings is canceled, so accuracy alone would mislead",
        "Share of bookings by outcome - stratified train and test sets",
        figsize=(9, 3.2),
    )
    sets = [("Train", data["train_rows"], data["train_cancel_rate"])]
    sets.append(("Test", data["test_rows"], data["test_cancel_rate"]))
    for y, (label, rows, rate) in enumerate(sets):
        ax.barh(y, 1 - rate, color=config.CLASS_COLORS[0], height=0.5)
        ax.barh(y, rate, left=1 - rate, color=config.CLASS_COLORS[1], height=0.5)
        ax.text(0.01, y, f"Not canceled {1 - rate:.1%}", va="center", color="white", fontsize=9)
        ax.text(0.99, y, f"Canceled {rate:.1%}", va="center", ha="right", color="white", fontsize=9)
        ax.text(-0.01, y, f"{label}\n{rows:,} rows", va="center", ha="right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.grid(visible=False)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    fig.subplots_adjust(left=0.14)
    return fig


def plot_cv_comparison(summary: dict[str, Any]) -> Figure:
    """Grouped bars of mean CV score (with fold std) per model and metric.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    models = _models(summary)
    metrics = ["average_precision", "roc_auc", "f1", "recall", "precision"]
    fig, (ax,) = new_figure(
        "Boosting and forests lead on every ranking metric; the baseline scores zero F1",
        f"Mean {config.CV_FOLDS}-fold CV score on the training set "
        "(error bar = 1 std across folds)",
        figsize=(12, 5.2),
    )
    width = 0.8 / len(models)
    x = np.arange(len(metrics))
    all_keys = [m["key"] for m in models]
    for i, model in enumerate(models):
        means = [model["cv"][m]["mean"] for m in metrics]
        stds = [model["cv"][m]["std"] for m in metrics]
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            means,
            width * 0.92,
            yerr=stds,
            color=model_color(model["key"], all_keys),
            label=model["name"],
            error_kw={"elinewidth": 0.8, "ecolor": PALETTE["ink_secondary"]},
        )
    ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("CV score")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.22)
    return fig


def _curve_figure(summary: dict[str, Any], kind: str) -> Figure:
    """ROC or precision-recall curves of every model on the test set.

    Args:
        summary: Run summary.
        kind: ``"roc"`` or ``"pr"``.

    Returns:
        The figure.
    """
    models = _models(summary)
    all_keys = [m["key"] for m in models]
    final_key = summary["final"]["key"]
    metric = "roc_auc" if kind == "roc" else "average_precision"
    leader = max(_models(summary, include_baseline=False), key=lambda m: m["test"][metric])
    if kind == "roc":
        title = f"Every model beats chance; {leader['name']} has the highest ROC-AUC"
        subtitle = "ROC curve on the held-out test set (area = ROC-AUC)"
        xlabel, ylabel = "False positive rate", "True positive rate (recall)"
    else:
        title = f"{leader['name']} keeps precision highest across recall levels"
        subtitle = "Precision-recall curve on the held-out test set (area = PR-AUC)"
        xlabel, ylabel = "Recall", "Precision"
    fig, (ax,) = new_figure(title, subtitle, figsize=(8, 6))
    for model in models:
        points = np.asarray(model["curves"][kind])
        is_final = model["key"] == final_key
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=model_color(model["key"], all_keys),
            linewidth=3 if is_final else 1.6,
            label=f"{model['name']} ({model['test'][metric]:.3f})"
            + ("  << selected" if is_final else ""),
            zorder=3 if is_final else 2,
        )
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=PALETTE["ink_muted"], linewidth=1, linestyle="--")
    else:
        base_rate = summary["dataset"]["test_cancel_rate"]
        ax.axhline(base_rate, color=PALETTE["ink_muted"], linewidth=1, linestyle="--")
        ax.annotate(
            f"base rate {base_rate:.1%}",
            (0.02, base_rate),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color=PALETTE["ink_muted"],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right" if kind == "roc" else "lower left")
    return fig


def plot_roc_curves(summary: dict[str, Any]) -> Figure:
    """ROC curves of all models (test set).

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    return _curve_figure(summary, "roc")


def plot_pr_curves(summary: dict[str, Any]) -> Figure:
    """Precision-recall curves of all models (test set).

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    return _curve_figure(summary, "pr")


def plot_confusion_matrix(summary: dict[str, Any]) -> Figure:
    """Confusion matrices of the final model at 0.5 and at the tuned threshold.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    final = summary["final"]
    (_, fp_default), (_, tp_default) = final["confusion_default"]
    (_, fp_tuned), (_, tp_tuned) = final["confusion_tuned"]
    if final["threshold"] > config.DEFAULT_THRESHOLD:
        fewer_alarms, fewer_catches = fp_default - fp_tuned, tp_default - tp_tuned
        trade = f"{fewer_alarms:,} fewer false alarms for {fewer_catches:,} fewer catches"
    else:
        more_catches, more_alarms = tp_tuned - tp_default, fp_tuned - fp_default
        trade = f"{more_catches:,} more catches for {more_alarms:,} more false alarms"
    fig, axes = new_figure(
        f"{final['name']}: the tuned threshold gives {trade}",
        "Confusion matrix on the held-out test set - default 0.5 threshold (left) vs tuned (right)",
        ncols=2,
        figsize=(11, 4.8),
    )
    cmap = LinearSegmentedColormap.from_list("cm", [PALETTE["bg_light"], PALETTE["primary"]])
    panels = [
        (config.DEFAULT_THRESHOLD, final["confusion_default"]),
        (final["threshold"], final["confusion_tuned"]),
    ]
    for ax, (threshold, matrix) in zip(axes, panels, strict=True):
        matrix = np.asarray(matrix)
        ax.imshow(matrix, cmap=cmap)
        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                dark = matrix[i, j] > matrix.max() * 0.55
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:,}\n{matrix[i, j] / total:.1%}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if dark else PALETTE["ink"],
                )
        ax.set_xticks([0, 1], [f"Pred. {c.lower()}" for c in config.CLASS_NAMES])
        ax.set_yticks([0, 1], [f"Actual {c.lower()}" for c in config.CLASS_NAMES])
        ax.grid(visible=False)
        ax.set_title(
            f"threshold {threshold:.2f}", loc="left", fontsize=10, color=PALETTE["ink_secondary"]
        )
    fig.subplots_adjust(left=0.13, wspace=0.45)
    return fig


def plot_overfitting_gaps(summary: dict[str, Any]) -> Figure:
    """Train vs CV vs test score per model (primary metric).

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    models = _models(summary)
    fig, (ax,) = new_figure(
        "Unlimited trees memorise the training data; boosting keeps the gap small",
        f"{PRIMARY_LABEL} on CV training folds, CV validation folds and the test set",
        figsize=(12, 5),
    )
    x = np.arange(len(models))
    series = [
        ("Train (CV folds)", "train", PALETTE["primary"]),
        ("Cross-validation", "cv", PALETTE["marigold_deep"]),
        ("Test", "test", PALETTE["teal_deep"]),
    ]
    width = 0.26
    for i, (label, source, color) in enumerate(series):
        values = [
            m["cv"][config.PRIMARY_METRIC]["mean"]
            if source == "cv"
            else m[source][config.PRIMARY_METRIC]
            for m in models
        ]
        bars = ax.bar(x + (i - 1) * width, values, width * 0.92, color=color, label=label)
        _label_bars(ax, bars, "{:.2f}")
    for xi, model in zip(x, models, strict=True):
        ax.annotate(
            model["diagnosis"]["label"],
            (xi, 1.06),
            ha="center",
            fontsize=8,
            color=PALETTE["pink"]
            if "verfit" in model["diagnosis"]["label"]
            else PALETTE["ink_secondary"],
        )
    ax.set_xticks(x, [m["name"].replace(" (", "\n(") for m in models], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(PRIMARY_LABEL)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(bottom=0.24)
    return fig


def plot_learning_curves(summary: dict[str, Any]) -> Figure:
    """Train and CV score vs training-set size for each model with a curve.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    curves = summary["learning_curves"]
    names = {m["key"]: m["name"] for m in summary["models"]}
    ncols = 2
    nrows = int(np.ceil(len(curves) / ncols))
    fig, axes = new_figure(
        "More data still helps the flexible models; the linear model has already plateaued",
        f"Learning curves - {PRIMARY_LABEL}, mean +/- 1 std over {config.CV_FOLDS} folds",
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 4 * nrows + 1),
    )
    for ax, (key, curve) in zip(axes, curves.items(), strict=False):
        sizes = np.asarray(curve["sizes"])
        for prefix, label, color in (
            ("train", "Train", PALETTE["primary"]),
            ("cv", "CV", PALETTE["pink"]),
        ):
            mean, std = np.asarray(curve[f"{prefix}_mean"]), np.asarray(curve[f"{prefix}_std"])
            ax.plot(sizes, mean, color=color, marker="o", markersize=4, label=label)
            ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_title(names[key], loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Training rows")
        ax.set_ylabel(PRIMARY_LABEL)
        ax.set_ylim(0.4, 1.02)
        ax.legend(loc="lower right")
    for ax in axes[len(curves) :]:
        ax.set_visible(False)
    return fig


def plot_validation_curve(summary: dict[str, Any]) -> Figure:
    """Decision Tree train vs CV score by ``max_depth``.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    curve = summary["validation_curve"]
    depths = np.asarray(curve["depths"])
    best = curve["best_depth"]
    fig, (ax,) = new_figure(
        f"Tree depth sweet spot is around {best}: shallower underfits, deeper overfits",
        f"Decision Tree validation curve - {PRIMARY_LABEL} by max_depth "
        f"({config.CV_FOLDS}-fold CV)",
    )
    for prefix, label, color in (
        ("train", "Train", PALETTE["primary"]),
        ("cv", "CV", PALETTE["pink"]),
    ):
        mean, std = np.asarray(curve[f"{prefix}_mean"]), np.asarray(curve[f"{prefix}_std"])
        ax.plot(depths, mean, color=color, marker="o", markersize=4, label=label)
        ax.fill_between(depths, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax.axvline(best, color=PALETTE["marigold_deep"], linewidth=1.5, linestyle="--")
    ax.annotate(
        f"best CV depth = {best}",
        (best, 0.45),
        xytext=(6, 0),
        textcoords="offset points",
        color=PALETTE["ink_secondary"],
        fontsize=9,
    )
    ax.axvspan(depths.min(), 3.5, color=PALETTE["primary"], alpha=0.05)
    ax.axvspan(best + 6, depths.max(), color=PALETTE["pink"], alpha=0.05)
    ax.text(1.2, 1.0, "underfitting", fontsize=8.5, color=PALETTE["ink_muted"])
    ax.text(
        depths.max() - 1, 1.0, "overfitting", fontsize=8.5, ha="right", color=PALETTE["ink_muted"]
    )
    ax.set_xlabel("max_depth")
    ax.set_ylabel(PRIMARY_LABEL)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right")
    return fig


def plot_threshold_tuning(summary: dict[str, Any]) -> Figure:
    """Precision, recall and F1 against the decision threshold (out-of-fold).

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    table = summary["threshold"]["oof"]
    chosen = summary["final"]["threshold"]
    thresholds = [row["threshold"] for row in table]
    fig, (ax,) = new_figure(
        f"Moving the threshold from 0.50 to {chosen:.2f} gives the best precision/recall balance",
        "Final model, out-of-fold predictions on the training set (the test set is not used)",
    )
    for metric, color in (
        ("precision", PALETTE["primary"]),
        ("recall", PALETTE["pink"]),
        ("f1", PALETTE["marigold_deep"]),
    ):
        ax.plot(
            thresholds, [row[metric] for row in table], color=color, label=METRIC_LABELS[metric]
        )
    ax.axvline(config.DEFAULT_THRESHOLD, color=PALETTE["ink_muted"], linewidth=1, linestyle=":")
    ax.axvline(chosen, color=PALETTE["teal_deep"], linewidth=2, linestyle="--")
    ax.annotate(
        f"chosen {chosen:.2f}",
        (chosen, 0.05),
        xytext=(5, 0),
        textcoords="offset points",
        color=PALETTE["ink_secondary"],
        fontsize=9,
    )
    ax.set_xlabel("Decision threshold (predict canceled if probability >= threshold)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    return fig


def _importance_bars(
    ax: Axes, labels: list[str], values: list[float], color: str, errors: list[float] | None = None
) -> None:
    """Horizontal importance bars, largest on top.

    Args:
        ax: Target axes.
        labels: Feature names.
        values: Importances.
        color: Bar colour.
        errors: Optional error bars.
    """
    bars = ax.barh(
        labels,
        values,
        xerr=errors,
        color=color,
        height=0.6,
        error_kw={"elinewidth": 0.8, "ecolor": PALETTE["ink_secondary"]},
    )
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _label_bars(ax, bars, "{:.3f}", horizontal=True)
    ax.margins(x=0.18)


def plot_permutation_importance(summary: dict[str, Any]) -> Figure:
    """Permutation importance of the final model (test set).

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    table = summary["permutation"][: config.TOP_FEATURES_TO_SHOW]
    top = [row["feature"] for row in table]
    fig, (ax,) = new_figure(
        f"{top[0]}, {top[1]} and {top[2]} drive the predictions most",
        f"Drop in test {PRIMARY_LABEL} when one input is shuffled "
        f"({config.PERMUTATION_REPEATS} repeats; final model: {summary['final']['name']})",
        figsize=(10, 6.5),
    )
    _importance_bars(
        ax,
        [r["feature"] for r in table],
        [r["importance_mean"] for r in table],
        PALETTE["pink"],
        [r["importance_std"] for r in table],
    )
    ax.set_xlabel(f"Mean drop in {PRIMARY_LABEL}")
    fig.subplots_adjust(left=0.27)
    return fig


def plot_tree_importances(summary: dict[str, Any]) -> Figure:
    """Impurity-based importances of the Decision Tree and the Random Forest.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    tables = summary["tree_importance"]
    fig, axes = new_figure(
        "Tree models agree with permutation importance on the main drivers",
        "Impurity-based feature importance, summed per input column (training fit)",
        ncols=len(tables),
        figsize=(13, 6),
    )
    colors = [PALETTE["marigold_deep"], PALETTE["teal_deep"]]
    for ax, (key, table), color in zip(axes, tables.items(), colors, strict=False):
        top = table[: config.TOP_FEATURES_TO_SHOW - 3]
        _importance_bars(ax, [r["feature"] for r in top], [r["importance"] for r in top], color)
        ax.set_title(config.MODEL_NAMES[base_key(key)], loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance (sums to 1)")
    fig.subplots_adjust(left=0.18, wspace=0.75)
    return fig


def plot_decision_tree(model: Pipeline, depth: int = config.FALLING_TREE_DEPTH) -> Figure:
    """Static ``plot_tree`` of the top levels of the fitted Decision Tree.

    Args:
        model: Fitted Decision Tree pipeline (tree preprocessing, raw units).
        depth: Levels to draw.

    Returns:
        The figure.
    """
    names_out = model.named_steps["prep"].get_feature_names_out()
    tree = model.named_steps["clf"].tree_
    first = [
        names_out[tree.feature[n]].split("__", 1)[-1]
        for n in (0, tree.children_left[0], tree.children_right[0])
        if tree.children_left[n] != tree.children_right[n]
    ]
    fig, (ax,) = new_figure(
        f"The tree splits first on {first[0]}, then on {' and '.join(dict.fromkeys(first[1:]))}",
        f"Top {depth} levels of the Decision Tree (class-weighted; orange = more canceled)",
        figsize=(16, 7.5),
    )
    names = [n.split("__", 1)[-1] for n in model.named_steps["prep"].get_feature_names_out()]
    artists = plot_tree(
        model.named_steps["clf"],
        max_depth=depth,
        feature_names=names,
        class_names=list(config.CLASS_NAMES),
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=2,
        fontsize=8,
        ax=ax,
    )
    for artist in artists:
        box = artist.get_bbox_patch()
        if box is not None:
            box.set_edgecolor(PALETTE["grid"])
    ax.grid(visible=False)
    return fig


def plot_temporal_vs_random(summary: dict[str, Any]) -> Figure:
    """Test score on the random split vs the latest-period (temporal) split.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    temporal = summary["temporal"]
    models = [
        m for m in _models(summary, include_baseline=False) if m["key"] in temporal["metrics"]
    ]
    fig, (ax,) = new_figure(
        "Predicting a later season is harder, and the ensembles lose most of their lead",
        f"Test {PRIMARY_LABEL}: random stratified split vs train before {temporal['cutoff']} / "
        f"test on later arrivals",
        figsize=(12, 4.8),
    )
    x = np.arange(len(models))
    random_scores = [m["test"][config.PRIMARY_METRIC] for m in models]
    temporal_scores = [temporal["metrics"][m["key"]][config.PRIMARY_METRIC] for m in models]
    bars = ax.bar(
        x - 0.2, random_scores, 0.38, color=PALETTE["primary"], label="Random split (test)"
    )
    _label_bars(ax, bars, "{:.2f}")
    bars = ax.bar(
        x + 0.2,
        temporal_scores,
        0.38,
        color=PALETTE["pink"],
        label="Temporal split (latest period)",
    )
    _label_bars(ax, bars, "{:.2f}")
    ax.set_xticks(x, [m["name"].replace(" (", "\n(") for m in models], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(PRIMARY_LABEL)
    ax.legend(loc="upper left", ncols=2)
    fig.subplots_adjust(bottom=0.17)
    return fig


def plot_speed_vs_score(summary: dict[str, Any]) -> Figure:
    """CV score against full-train fit time; bubble area = model file size.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    models = _models(summary, include_baseline=False)
    all_keys = [m["key"] for m in summary["models"]]
    fig, (ax,) = new_figure(
        "Accuracy vs cost: the selected model scores at the top with a small file",
        f"CV {PRIMARY_LABEL} vs fit time on the full training set (bubble area = file size)",
    )
    sizes = np.asarray([m["size_mb"] for m in models])
    area = 60 + 1400 * np.sqrt(sizes / max(sizes.max(), 1e-9))
    for model, s in zip(models, area, strict=True):
        ax.scatter(
            model["fit_time_s"],
            model["cv"][config.PRIMARY_METRIC]["mean"],
            s=s,
            color=model_color(model["key"], all_keys),
            alpha=0.75,
            edgecolor="white",
        )
        ax.annotate(
            f"{model['name']}\n{model['size_mb']:.1f} MB",
            (model["fit_time_s"], model["cv"][config.PRIMARY_METRIC]["mean"]),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=8,
            color=PALETTE["ink_secondary"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Fit time on the full training set (s, log scale)")
    ax.set_ylabel(f"CV {PRIMARY_LABEL}")
    ax.margins(x=0.25, y=0.15)
    return fig


def plot_leakage_ablation(summary: dict[str, Any]) -> Figure:
    """CV score of the final model with suspected post-booking columns added/removed.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    rows = summary["ablation"]
    fig, (ax,) = new_figure(
        "Post-booking columns would add a suspicious boost, so they stay out",
        f"CV {PRIMARY_LABEL} of the final model with each input change (training set)",
        figsize=(10, 4.4),
    )
    colors = [
        PALETTE["marigold_deep"]
        if r["delta"] == 0
        else PALETTE["pink"]
        if r["delta"] > 0
        else PALETTE["primary"]
        for r in rows
    ]
    bars = ax.barh(
        [r["variant"] for r in rows], [r["pr_auc"] for r in rows], color=colors, height=0.55
    )
    for bar, row in zip(bars, rows, strict=True):
        ax.annotate(
            f"{row['pr_auc']:.4f}  ({row['delta']:+.4f})",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=PALETTE["ink_secondary"],
        )
    low = min(r["pr_auc"] for r in rows)
    ax.set_xlim(low - 0.05, 1.0)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"CV {PRIMARY_LABEL}")
    fig.subplots_adjust(left=0.27)
    return fig


def plot_boosting_replay(summary: dict[str, Any]) -> Figure:
    """HistGradientBoosting train and validation score after each boosting stage.

    Args:
        summary: Run summary.

    Returns:
        The figure.
    """
    replay = summary["replay"]
    fig, (ax,) = new_figure(
        f"Validation score climbs fast, then flattens near iteration {replay['best_iteration']}",
        f"HistGradientBoosting {PRIMARY_LABEL} by boosting stage (validation = 20% of train)",
    )
    ax.plot(replay["iterations"], replay["train"], color=PALETTE["primary"], label="Train")
    ax.plot(replay["iterations"], replay["validation"], color=PALETTE["pink"], label="Validation")
    ax.axvline(
        replay["best_iteration"], color=PALETTE["marigold_deep"], linestyle="--", linewidth=1.5
    )
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel(PRIMARY_LABEL)
    ax.legend(loc="lower right")
    return fig


def generate_figures(summary: dict[str, Any], tree_model: Pipeline) -> list[Path]:
    """Create and save every evaluation figure.

    Args:
        summary: Run summary built by ``main.py``.
        tree_model: Fitted Decision Tree pipeline (for the static tree plot).

    Returns:
        Paths of the saved figures, in order.
    """
    figures = {
        "01_class_balance": plot_class_balance(summary),
        "02_cv_model_comparison": plot_cv_comparison(summary),
        "03_roc_curves": plot_roc_curves(summary),
        "04_pr_curves": plot_pr_curves(summary),
        "05_confusion_matrix": plot_confusion_matrix(summary),
        "06_overfitting_gaps": plot_overfitting_gaps(summary),
        "07_learning_curves": plot_learning_curves(summary),
        "08_validation_curve_depth": plot_validation_curve(summary),
        "09_threshold_tuning": plot_threshold_tuning(summary),
        "10_permutation_importance": plot_permutation_importance(summary),
        "11_tree_feature_importance": plot_tree_importances(summary),
        "12_decision_tree_top_levels": plot_decision_tree(tree_model),
        "13_temporal_vs_random": plot_temporal_vs_random(summary),
        "14_speed_vs_score": plot_speed_vs_score(summary),
        "15_leakage_ablation": plot_leakage_ablation(summary),
        "16_boosting_replay": plot_boosting_replay(summary),
    }
    return [save_figure(fig, name) for name, fig in figures.items()]
