"""Shared fixtures: a small, realistic sample of cleaned + engineered bookings."""

import pandas as pd
import pytest

from src import config
from src.data import clean_data, load_raw_data
from src.features import FeatureGroups, engineer_features

SAMPLE_ROWS: int = 3_000


@pytest.fixture(scope="session")
def featured_sample() -> pd.DataFrame:
    """3,000 real raw bookings (seeded sample), cleaned and engineered like in main.py."""
    raw = load_raw_data().sample(SAMPLE_ROWS, random_state=config.RANDOM_STATE)
    return engineer_features(clean_data(raw).data).reset_index(drop=True)


@pytest.fixture(scope="session")
def groups() -> FeatureGroups:
    """The final model inputs."""
    return FeatureGroups.selected()


@pytest.fixture(scope="session")
def xy(featured_sample: pd.DataFrame, groups: FeatureGroups) -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs and labels of the sample."""
    return featured_sample[groups.all], featured_sample[config.TARGET]
