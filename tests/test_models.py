"""Tests for the model zoo in src.models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import config, models
from src.features import FeatureGroups


def test_zoo_has_baseline_and_five_algorithms(groups: FeatureGroups) -> None:
    """Every configured algorithm is in the zoo, baseline included."""
    zoo = models.build_model_zoo(groups)
    assert list(zoo) == ["dummy", "logreg", "tree", "rf", "hgb", "knn"]
    assert all(list(p.named_steps) == ["prep", "clf"] for p in zoo.values())


@pytest.mark.parametrize("key", list(models.MODEL_SPECS))
def test_every_pipeline_fits_and_predicts_probabilities(
    key: str, groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Each pipeline trains on real rows and returns valid probabilities."""
    x, y = xy
    pipeline = models.build_pipeline(key, groups)
    if key == "rf":
        pipeline.set_params(clf__n_estimators=20)
    proba = pipeline.fit(x, y).predict_proba(x.head(50))
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_scaled_vs_tree_preprocessing(groups: FeatureGroups) -> None:
    """Linear/distance models are scaled; tree models are not."""
    for key, spec in models.MODEL_SPECS.items():
        prep = models.build_pipeline(key, groups).named_steps["prep"]
        numeric = prep.transformers[0][1]
        is_scaled = hasattr(numeric, "named_steps") and "scale" in numeric.named_steps
        assert is_scaled == (spec.preprocessing == "scaled"), key


def test_param_distributions_are_valid(groups: FeatureGroups) -> None:
    """Every search-space parameter exists on its pipeline (catches config typos)."""
    for key, space in config.PARAM_DISTRIBUTIONS.items():
        pipeline = models.build_pipeline(key, groups)
        for name, values in space.items():
            clone(pipeline).set_params(**{name: values[0]})


def test_overrides_and_seed() -> None:
    """Overrides replace configured values; seeded models get the project seed."""
    tree = models.make_estimator("tree", max_depth=3)
    assert tree.max_depth == 3 and tree.random_state == config.RANDOM_STATE
    assert models.make_estimator("hgb_tuned").class_weight == "balanced"


def test_names_and_keys() -> None:
    """Tuned keys map back to their algorithm and get a "(tuned)" label."""
    assert models.base_key("rf_tuned") == "rf"
    assert models.display_name("rf_tuned") == "Random Forest (tuned)"
    assert models.display_name("logreg") == "Logistic Regression"


def test_model_size_is_positive(groups: FeatureGroups, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    """A fitted model has a measurable compressed size."""
    fitted = models.build_pipeline("tree", groups, {"max_depth": 3}).fit(*xy)
    assert 0 < models.model_size_mb(fitted) < 5
