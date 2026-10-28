"""Tests for the Flask web app: pages, data API, prediction API and health check.

They use the committed caches and final model, exactly as the deployed server does.
"""

import random

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.app import create_app
from app.services import (
    DASHBOARD_SECTIONS,
    EXPLORE_SECTIONS,
    AppData,
    Predictor,
    load_app_data,
    random_sample,
    validate_booking,
)


@pytest.fixture(scope="module")
def app_data() -> AppData:
    """App data loaded from the caches, as on the server."""
    return load_app_data()


@pytest.fixture(scope="module")
def app(app_data: AppData) -> Flask:
    """App instance using the shared data."""
    flask_app = create_app(app_data)
    flask_app.testing = True
    return flask_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Test client."""
    return app.test_client()


@pytest.fixture
def booking(app_data: AppData) -> dict:
    """The typical booking from the cache."""
    return dict(app_data.predict["defaults"])


@pytest.mark.parametrize(
    "path", ["/", "/models", "/overfitting", "/explainability", "/predict", "/healthz"]
)
def test_pages_load(client: FlaskClient, path: str) -> None:
    """Every page renders successfully."""
    assert client.get(path).status_code == 200


def test_overview_shows_real_numbers(client: FlaskClient, app_data: AppData) -> None:
    """The overview shows the final model and the dataset sizes."""
    html = client.get("/").get_data(as_text=True)
    assert app_data.dashboard["final"]["name"] in html
    assert "87,208" in html and "119,390" in html


def test_unknown_page_is_404(client: FlaskClient) -> None:
    """Unknown URLs get the custom 404 page."""
    response = client.get("/nope")
    assert response.status_code == 404 and "canceled" in response.get_data(as_text=True)


@pytest.mark.parametrize("section", [*DASHBOARD_SECTIONS, *EXPLORE_SECTIONS, "predict"])
def test_data_sections(client: FlaskClient, section: str) -> None:
    """Every data section is served as non-empty JSON."""
    response = client.get(f"/api/data/{section}")
    assert response.status_code == 200 and response.get_json()


def test_unknown_data_section(client: FlaskClient) -> None:
    """Unknown sections return a JSON 404."""
    response = client.get("/api/data/secret")
    assert response.status_code == 404 and "error" in response.get_json()


def test_models_have_complete_records(client: FlaskClient) -> None:
    """Every model has CV, train and test metrics, curves and a diagnosis."""
    models = client.get("/api/data/models").get_json()
    assert len(models) >= 6
    for model in models:
        assert {"cv", "train", "test", "curves", "confusion", "diagnosis", "size_mb"} <= set(model)


def test_predict_returns_probability_and_explanation(client: FlaskClient, booking: dict) -> None:
    """A valid booking gets a probability, verdict, contributions and sweeps."""
    result = client.post("/api/predict", json=booking).get_json()
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] == int(result["probability"] >= result["threshold"])
    assert result["risk"] in {"low", "medium", "high"}
    assert set(result["sweeps"]) == {"lead_time", "adr"}
    assert result["contributions"] == []  # the typical booking differs from itself nowhere


def test_risky_booking_scores_higher(client: FlaskClient, booking: dict) -> None:
    """Long lead time + previous cancellations + no requests raises the risk."""
    safe = client.post("/api/predict", json=booking).get_json()["probability"]
    risky = {
        **booking,
        "lead_time": 400,
        "previous_cancellations": 3,
        "total_of_special_requests": 0,
    }
    result = client.post("/api/predict", json=risky).get_json()
    assert result["probability"] > safe
    assert {c["field"] for c in result["contributions"]} >= {"lead_time", "previous_cancellations"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"lead_time": 9999}, "between"),
        ({"adr": "cheap"}, "number"),
        ({"hotel": "Motel"}, "Unknown value"),
        ({"arrival_date": "31/02/2017"}, "YYYY-MM-DD"),
        ({"adults": 1, "children": 0, "is_repeated_guest": 5}, "0 or 1"),
    ],
)
def test_predict_rejects_bad_input(
    client: FlaskClient, booking: dict, change: dict, message: str
) -> None:
    """Invalid values get a clear 400 error."""
    response = client.post("/api/predict", json={**booking, **change})
    assert response.status_code == 400 and message in response.get_json()["error"]


def test_predict_rejects_missing_fields_and_non_json(client: FlaskClient) -> None:
    """Missing fields and non-JSON bodies are rejected."""
    assert (
        "Missing fields"
        in client.post("/api/predict", json={"hotel": "City Hotel"}).get_json()["error"]
    )
    assert client.post("/api/predict", data="x", content_type="text/plain").status_code == 400


def test_sample_endpoint_returns_valid_booking(client: FlaskClient, app_data: AppData) -> None:
    """Random samples are real bookings that pass validation and carry the outcome."""
    sample = client.get("/api/sample").get_json()
    assert sample["actual"] in (0, 1)
    validate_booking(sample, app_data.predict["fields"])


def test_every_cached_sample_can_be_predicted(app_data: AppData) -> None:
    """All cached sample bookings are valid inputs for the model."""
    predictor = Predictor(app_data)
    rng = random.Random(0)
    for _ in range(25):
        booking = validate_booking(random_sample(app_data, rng), app_data.predict["fields"])
        assert 0 <= predictor.predict(booking)["probability"] <= 1


def test_pages_carry_data_version(client: FlaskClient, app_data: AppData) -> None:
    """Every page tells the front end which cache version to request (cache busting)."""
    assert len(app_data.version) == 10
    assert f'data-version="{app_data.version}"' in client.get("/").get_data(as_text=True)
