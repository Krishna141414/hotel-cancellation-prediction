"""Tests for the interactive exports in src.insights."""

import numpy as np
import pandas as pd
import pytest

from src import config, insights
from src.features import FeatureGroups, split_frame
from src.models import build_pipeline


@pytest.fixture(scope="module")
def tree_model(groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]) -> object:
    """A small fitted Decision Tree pipeline."""
    return build_pipeline("tree", groups, {"max_depth": 5}).fit(*xy)


def _walk(node: dict) -> list[dict]:
    """Flatten a nested tree export."""
    return [node] + [n for child in node.get("children", []) for n in _walk(child)]


def test_export_tree_counts_are_consistent(
    tree_model: object, groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Children split their parent's rows exactly and depth is limited."""
    x, y = xy
    root = insights.export_tree(tree_model, groups, x, y, max_depth=3)
    assert root["rows"] == len(x)
    assert root["cancel_rate"] == pytest.approx(y.mean(), abs=1e-3)
    for node in _walk(root):
        if "children" in node:
            assert sum(c["rows"] for c in node["children"]) == node["rows"]
            assert {c["branch"] for c in node["children"]} == {"yes", "no"}
    assert len(_walk(root)) <= 15


def test_split_question_is_readable(
    groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Category splits ask 'column = value?'; integer columns show whole numbers."""
    x, _ = xy
    question = insights.split_question("cat__deposit_type_Non Refund", 0.5, groups, x)
    assert question == {"question": "deposit_type = Non Refund?", "left": "no", "right": "yes"}
    assert (
        insights.split_question("num__lead_time", 12.5, groups, x)["question"] == "lead_time <= 12?"
    )


def test_falling_paths_follow_the_tree(
    tree_model: object, groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Every path starts at the root, has at most depth+1 nodes and ends in a leaf class."""
    x, y = xy
    result = insights.falling_paths(tree_model, groups, x, y, x.head(40), y.head(40), depth=3)
    assert len(result["samples"]) == 40
    assert all(s["path"][0] == 0 and len(s["path"]) <= 4 for s in result["samples"])
    assert 0 <= result["accuracy"] <= 1


def test_decision_boundaries_shape(
    monkeypatch: pytest.MonkeyPatch, featured_sample: pd.DataFrame
) -> None:
    """Each pair/model has a full grid of 0-100 values and train/test scores."""
    monkeypatch.setattr(config, "PLAYGROUND_SAMPLE", 800)
    monkeypatch.setattr(config, "PLAYGROUND_POINTS", 50)
    monkeypatch.setattr(config, "PLAYGROUND_GRID", 12)
    train_rows, test_rows = split_frame(featured_sample)
    result = insights.decision_boundaries(train_rows, test_rows)
    assert len(result["pairs"]) == len(config.PLAYGROUND_PAIRS)
    for pair in result["pairs"]:
        assert len(pair["points"]) == 50
        for scores in pair["results"].values():
            grid = np.asarray(scores["grid"])
            assert grid.shape == (144,) and grid.min() >= 0 and grid.max() <= 100


def test_pca_projection(
    monkeypatch: pytest.MonkeyPatch, groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Three components, variance shares in (0, 1), points labelled 0/1."""
    monkeypatch.setattr(config, "PCA_SAMPLES", 300)
    result = insights.pca_projection(*xy, groups)
    points = np.asarray(result["points"])
    assert points.shape == (300, 4) and set(points[:, 3]) <= {0.0, 1.0}
    assert 0 < sum(result["explained"]) < 1


def test_boosting_replay(groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    """One score per kept stage; the best iteration is within the run."""
    result = insights.boosting_replay({"max_iter": 40}, groups, *xy)
    assert len(result["train"]) == len(result["validation"]) == len(result["iterations"])
    assert 1 <= result["best_iteration"] <= 40


def test_feature_insights(featured_sample: pd.DataFrame, groups: FeatureGroups) -> None:
    """Numeric, binary and categorical columns all get a sentence with rates."""
    items = insights.feature_insights(
        featured_sample, ["lead_time", "is_domestic", "deposit_type"], groups
    )
    assert [i["feature"] for i in items] == ["lead_time", "is_domestic", "deposit_type"]
    assert all(0 <= i["low_rate"] <= 1 and 0 <= i["high_rate"] <= 1 and i["text"] for i in items)
