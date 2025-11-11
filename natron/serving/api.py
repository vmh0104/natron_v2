from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

from ..config import DEFAULT_CONFIG_PATH
from .service import InferenceConfig, NatronInferenceService


def create_app(model_path: str | None = None, config_path: str | None = None) -> Flask:
    model_path = model_path or os.environ.get("NATRON_MODEL_PATH", "model/natron_v2.pt")
    config_path = config_path or os.environ.get("NATRON_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    inference_config = InferenceConfig(model_path=Path(model_path), config_path=Path(config_path))
    service = NatronInferenceService(inference_config)

    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return {"status": "ok"}

    @app.post("/predict")
    def predict() -> Any:
        payload: Dict[str, Any] = request.get_json(force=True)
        candles = payload.get("candles")
        if not candles:
            return jsonify({"error": "Missing 'candles' field"}), 400
        try:
            result = service.predict_from_candles(candles)
        except Exception as exc:  # pragma: no cover - runtime error propagation
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    return app


def run_app(model_path: str | None = None, config_path: str | None = None, host: str = "0.0.0.0", port: int = 8000) -> None:
    app = create_app(model_path, config_path)
    app.run(host=host, port=port)
