"""Flask application: five pages, a small JSON API and a health check.

Run locally:   flask --app app.app run      (or: python -m app.app)
Production:    gunicorn app.app:app         (settings in gunicorn.conf.py)
"""

import logging
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

from app.services import AppData, Predictor, load_app_data, random_sample, validate_booking
from src import config

logger = logging.getLogger(__name__)

NAV_PAGES: list[tuple[str, str]] = [
    ("overview", "Overview"),
    ("models", "Model comparison"),
    ("overfitting", "Overfitting"),
    ("explainability", "Explainability"),
    ("predict", "Predict"),
]


def _error(message: str, status: int = 400) -> tuple[Response, int]:
    """Build a JSON error response.

    Args:
        message: Human-readable explanation.
        status: HTTP status code.

    Returns:
        ``(response, status)``.
    """
    return jsonify({"error": message}), status


def create_app(data: AppData | None = None) -> Flask:
    """Create the Flask app.

    Args:
        data: Pre-loaded app data (tests pass their own); loaded from disk if ``None``.

    Returns:
        The configured Flask application.
    """
    config.setup_logging()
    # Feature engineering logs every call; per-request logs would only be noise here.
    logging.getLogger("src.features").setLevel(logging.WARNING)
    app = Flask(__name__)
    app.json.sort_keys = False
    data = data or load_app_data()
    predictor = Predictor(data)

    @app.context_processor
    def inject_layout() -> dict[str, Any]:
        """Navigation and headline numbers available to every template."""
        return {
            "nav_pages": NAV_PAGES,
            "final": data.dashboard["final"],
            "final_model": data.final_model,
            "dataset": data.dashboard["dataset"],
            "setup": data.dashboard["setup"],
        }

    # -- Pages ----------------------------------------------------------------
    @app.get("/")
    def overview() -> str:
        """Problem, dataset, final model and headline metrics."""
        return render_template("overview.html", page="overview", data=data.dashboard)

    @app.get("/models")
    def models() -> str:
        """Leaderboard, comparison table and charts."""
        return render_template("models.html", page="models")

    @app.get("/overfitting")
    def overfitting() -> str:
        """Train vs CV vs test, learning/validation curves, playground, replay, drift."""
        return render_template("overfitting.html", page="overfitting")

    @app.get("/explainability")
    def explainability() -> str:
        """Feature importance, interactive tree, falling bookings and the 3-D space."""
        return render_template("explainability.html", page="explainability")

    @app.get("/predict")
    def predict() -> str:
        """Booking form, risk gauge, what-if simulator and threshold tools."""
        return render_template("predict.html", page="predict")

    # -- API ------------------------------------------------------------------
    @app.get("/api/data/<section>")
    def api_data(section: str) -> Response | tuple[Response, int]:
        """One precomputed data section (cached by the browser)."""
        try:
            payload = data.section(section)
        except KeyError:
            return _error(f"Unknown data section: {section}", 404)
        response = jsonify(payload)
        response.cache_control.public = True
        response.cache_control.max_age = 3600
        return response

    @app.post("/api/predict")
    def api_predict() -> Response | tuple[Response, int]:
        """Cancellation probability + explanation for one booking."""
        try:
            booking = validate_booking(request.get_json(silent=True), data.predict["fields"])
        except ValueError as error:
            return _error(str(error))
        return jsonify(predictor.predict(booking))

    @app.get("/api/sample")
    def api_sample() -> Response:
        """A random real test booking (with its true outcome)."""
        return jsonify(random_sample(data))

    @app.get("/healthz")
    def healthz() -> tuple[str, int]:
        """Health check for the hosting platform."""
        return "ok", 200

    @app.errorhandler(404)
    def not_found(_: Exception) -> tuple[str, int]:
        """Friendly 404 page."""
        return render_template("404.html", page=""), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False)
