"""Download and verify the raw Hotel Booking Demand dataset.

Run directly with ``python -m src.download_data``. ``main.py`` also calls
:func:`ensure_raw_data` so the project works from scratch on any machine.
"""

import logging
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, timeout: int) -> None:
    """Download a URL to a local file.

    Args:
        url: Source URL.
        destination: File path to write to (parent folder must exist).
        timeout: Network timeout in seconds.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        destination.write_bytes(response.read())


def verify_shape(csv_path: Path, expected_rows: int, expected_columns: int) -> None:
    """Check that a CSV has exactly the expected number of rows and columns.

    Args:
        csv_path: Path to the CSV file.
        expected_rows: Required number of data rows (header excluded).
        expected_columns: Required number of columns.

    Raises:
        ValueError: If the shape does not match.
    """
    rows, columns = pd.read_csv(csv_path, low_memory=False).shape
    if (rows, columns) != (expected_rows, expected_columns):
        raise ValueError(
            f"Unexpected shape {rows:,} x {columns}; "
            f"expected {expected_rows:,} x {expected_columns}."
        )


def download_dataset(
    urls: tuple[str, ...] = config.DATA_URLS,
    destination: Path = config.RAW_DATA_PATH,
) -> Path:
    """Try each mirror URL in order until one downloads and verifies.

    The file is first written to a temporary ``.part`` file and only renamed
    to ``destination`` after verification, so a failed download never leaves
    a broken raw file behind.

    Args:
        urls: Candidate URLs, tried in order (primary first, then fallbacks).
        destination: Final path of the raw CSV.

    Returns:
        Path to the verified raw CSV.

    Raises:
        RuntimeError: If every URL fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    for url in urls:
        logger.info("Downloading dataset from %s", url)
        try:
            download_file(url, temp_path, config.DOWNLOAD_TIMEOUT_SECONDS)
            verify_shape(temp_path, config.EXPECTED_ROWS, config.EXPECTED_COLUMNS)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            logger.warning("Failed: %s", error)
            temp_path.unlink(missing_ok=True)
            continue

        temp_path.replace(destination)
        size_mb = destination.stat().st_size / 1_000_000
        logger.info(
            "SUCCESS: saved %s (%.1f MB, %s rows x %s columns verified)",
            destination.relative_to(config.PROJECT_ROOT),
            size_mb,
            f"{config.EXPECTED_ROWS:,}",
            config.EXPECTED_COLUMNS,
        )
        return destination

    raise RuntimeError("FAILURE: could not download the dataset from any mirror URL.")


def ensure_raw_data(path: Path = config.RAW_DATA_PATH) -> Path:
    """Return the raw CSV path, downloading the dataset first if it is missing.

    Args:
        path: Expected location of the raw CSV.

    Returns:
        Path to the raw CSV.
    """
    if path.exists():
        logger.info(
            "Raw data found at %s - skipping download", path.relative_to(config.PROJECT_ROOT)
        )
        return path
    logger.info("Raw data not found - downloading")
    return download_dataset(destination=path)


if __name__ == "__main__":
    config.setup_logging()
    ensure_raw_data()
