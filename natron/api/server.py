from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from flask import Flask, jsonify, request

from natron.config.loader import load_natron_config
from natron.data.feature_engine import FeatureEngine
from natron.data.sequence import SequenceCreator
from natron.models.transformer import NatronTransformer
from natron.utils.logging import configure_logging
from natron.utils.torch_utils import get_device, load_checkpoint

app = Flask(__name__)
configure_logging()

_MODEL: NatronTransformer | None = None
_DEVICE = get_device()
_CONFIG = load_natron_config()
_REGIME_MAP = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE",
}


def load_model() -> NatronTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_path = Path(_CONFIG.get("paths")["supervised_model"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    checkpoint, _ = load_checkpoint(model_path, map_location=_DEVICE)
    input_dim = 100  # FeatureEngine outputs ~100 features
    model = NatronTransformer(input_dim=input_dim, model_cfg=_CONFIG.get("model"))
    model.load_state_dict(checkpoint["model_state"])
    model.to(_DEVICE)
    model.eval()
    _MODEL = model
    return model


def preprocess_candles(candles: List[Dict[str, float]]) -> torch.Tensor:
    import pandas as pd

    df = pd.DataFrame(candles)
    if "time" not in df.columns:
        df["time"] = np.arange(len(df))
    features = FeatureEngine().transform(df)
    features = features.tail(_CONFIG.get("data")["sequence_length"])
    sequences, _ = SequenceCreator(sequence_length=_CONFIG.get("data")["sequence_length"]).create(features, pd.DataFrame({
        "buy": np.zeros(len(features)),
        "sell": np.zeros(len(features)),
        "direction": np.zeros(len(features)),
        "regime": np.zeros(len(features)),
    }))
    return torch.from_numpy(sequences[-1:]).float().to(_DEVICE)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    candles = payload.get("candles", [])
    if len(candles) < _CONFIG.get("data")["sequence_length"]:
        return jsonify({"error": "Need at least 96 candles"}), 400

    model = load_model()
    with torch.no_grad():
        inputs = preprocess_candles(candles)
        outputs, _ = model(inputs)
        buy_prob = torch.sigmoid(outputs["buy"])[0].item()
        sell_prob = torch.sigmoid(outputs["sell"])[0].item()
        direction_probs = torch.softmax(outputs["direction"], dim=-1)[0].cpu().numpy()
        regime_probs = torch.softmax(outputs["regime"], dim=-1)[0].cpu().numpy()

        regime_id = int(np.argmax(regime_probs))
        confidence = float(np.max(regime_probs))

    response = {
        "buy_prob": round(buy_prob, 4),
        "sell_prob": round(sell_prob, 4),
        "direction_up": round(float(direction_probs[1]), 4),
        "regime": _REGIME_MAP[regime_id],
        "confidence": round(confidence, 4),
    }
    return jsonify(response)


def create_app():
    load_model()
    return app


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8000, debug=False)
