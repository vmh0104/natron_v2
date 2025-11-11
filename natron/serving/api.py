"""Flask API serving Natron predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request

from natron.pipelines.inference import (
    DIRECTION_NAMES,
    REGIME_NAMES,
    NatronPredictor,
    load_predictor,
)
from natron.utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


def create_app(model_path: Path, device: Optional[str] = None) -> Flask:
    predictor = load_predictor(model_path, device)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> tuple[dict, int]:
        return {"status": "ok"}, 200

    @app.route("/predict", methods=["POST"])
    def predict() -> tuple[dict, int]:
        try:
            payload = request.get_json(force=True)
            if payload is None or "candles" not in payload:
                return {"error": "Request must include 'candles' payload."}, 400
            candles = payload["candles"]
            prediction = predictor.predict(candles)
            return jsonify(prediction), 200
        except ValueError as exc:
            logger.exception("Validation error: %s", exc)
            return {"error": str(exc)}, 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled exception during prediction")
            return {"error": "Internal server error"}, 500

    @app.route("/model-info", methods=["GET"])
    def model_info() -> tuple[dict, int]:
        info = {
            "feature_count": len(predictor.bundle.feature_columns),
            "window_length": predictor.bundle.window_length,
            "device": str(predictor.bundle.device),
            "regime_labels": list(REGIME_NAMES.values()),
            "direction_labels": list(DIRECTION_NAMES.values()),
            "feature_columns": predictor.bundle.feature_columns,
        }
        return jsonify(info), 200

    return app


def run_app(model_path: Path, host: str = "0.0.0.0", port: int = 8000, device: Optional[str] = None) -> None:
    setup_logging(Path("logs"), "natron_api")
    app = create_app(model_path, device)
    app.run(host=host, port=port)


__all__ = ["create_app", "run_app"]
