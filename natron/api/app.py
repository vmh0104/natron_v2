"""Flask API for Natron Transformer predictions."""

from __future__ import annotations

from flask import Flask, jsonify, request

from natron.api.predictor import REGIME_MAP, NatronPredictor


def build_app(config_path: str | None = None) -> Flask:
    predictor = NatronPredictor(config_path)
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(force=True)
        candles = payload.get("candles")
        if candles is None or not isinstance(candles, list):
            return jsonify({"error": "Payload must include 'candles' list"}), 400
        try:
            result = predictor.predict(candles)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        response = {
            "buy_prob": round(result["buy_prob"], 4),
            "sell_prob": round(result["sell_prob"], 4),
            "direction_up": round(float(result["direction_probs"][1]), 4),
            "direction_neutral": round(float(result["direction_probs"][2]), 4) if len(result["direction_probs"]) > 2 else None,
            "direction_down": round(float(result["direction_probs"][0]), 4),
            "regime": result["regime_label"],
            "confidence": round(result["confidence"], 4),
            "regime_probs": {REGIME_MAP[i]: round(float(p), 4) for i, p in enumerate(result["regime_probs"])},
        }
        return jsonify(response)

    return app
