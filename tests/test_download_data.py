"""Tests for src.download_data, using local file:// URLs instead of the network."""

from pathlib import Path

import pandas as pd
import pytest

from src import download_data


@pytest.fixture
def source_csv(tmp_path: Path) -> Path:
    """A 3 x 2 CSV that stands in for the remote dataset."""
    path = tmp_path / "source" / "hotels.csv"
    path.parent.mkdir()
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(path, index=False)
    return path


def test_verify_shape_accepts_expected(source_csv: Path) -> None:
    """The right shape passes silently."""
    download_data.verify_shape(source_csv, expected_rows=3, expected_columns=2)


def test_verify_shape_rejects_wrong_shape(source_csv: Path) -> None:
    """A wrong shape raises a clear error."""
    with pytest.raises(ValueError, match="Unexpected shape"):
        download_data.verify_shape(source_csv, expected_rows=119_390, expected_columns=32)


def test_download_falls_back_to_second_url(
    source_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken primary URL is skipped and the fallback is used."""
    monkeypatch.setattr(download_data.config, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(download_data.config, "EXPECTED_COLUMNS", 2)
    monkeypatch.setattr(download_data.config, "PROJECT_ROOT", tmp_path)
    destination = tmp_path / "raw" / "hotel_bookings.csv"
    urls = ((tmp_path / "missing.csv").as_uri(), source_csv.as_uri())

    result = download_data.download_dataset(urls=urls, destination=destination)

    assert result == destination and destination.exists()
    assert not destination.with_suffix(".csv.part").exists()


def test_download_fails_cleanly_when_all_urls_fail(tmp_path: Path) -> None:
    """No partial file is left behind when every URL fails."""
    destination = tmp_path / "raw" / "hotel_bookings.csv"
    with pytest.raises(RuntimeError, match="FAILURE"):
        download_data.download_dataset(
            urls=((tmp_path / "nope.csv").as_uri(),), destination=destination
        )
    assert not destination.exists()
    assert not destination.with_suffix(".csv.part").exists()


def test_ensure_raw_data_skips_existing_file(
    source_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing file is returned without downloading."""
    monkeypatch.setattr(download_data.config, "PROJECT_ROOT", source_csv.parent.parent)

    def fail(**_: object) -> Path:
        raise AssertionError("should not download")

    monkeypatch.setattr(download_data, "download_dataset", fail)
    assert download_data.ensure_raw_data(source_csv) == source_csv
