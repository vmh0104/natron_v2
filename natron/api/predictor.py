"""Reusable prediction utilities for Natron inference services."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from natron.config import load_config
from natron.features.engine import FeatureEngine
from natron.models.transformer import NatronTransformer
from natron.utils import create_logger

REGIME_MAP = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE",
}


class NatronPredictor:
    def __init__(self, config_path: str | None = None):
        self.cfg = load_config(config_path)
        self.logger = create_logger("natron.predictor", self.cfg.training.log_dir)
        self.device = torch.device(
            self.cfg.api.device if torch.cuda.is_available() and self.cfg.api.device == "cuda" else "cpu"
        )
        self.model_path = Path(self.cfg.api.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")

        self.scaler_path = self.model_path.with_suffix(".scaler.npz")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}")

        self.logger.info("Loading model from %s", self.model_path)
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.feature_engine = FeatureEngine(self.cfg.features)

    def _load_model(self) -> NatronTransformer:
        state = torch.load(self.model_path, map_location="cpu")
        feature_dim = None
        for key, val in state.items():
            if key.endswith("input_projection.weight"):
                feature_dim = val.shape[1]
                break
        if feature_dim is None:
            raise RuntimeError("Unable to infer feature dimension from checkpoint.")
        model = NatronTransformer(feature_dim=feature_dim, config=self.cfg.model)
        incompat = model.load_state_dict(state, strict=False)
        if incompat.missing_keys:
            self.logger.warning("Missing keys while loading model: %s", incompat.missing_keys)
        if incompat.unexpected_keys:
            self.logger.warning("Unexpected keys while loading model: %s", incompat.unexpected_keys)
        model.to(self.device)
        model.eval()
        return model

    def _load_scaler(self) -> Dict[str, np.ndarray]:
        data = np.load(self.scaler_path)
        return {"mean": data["mean"], "std": data["std"]}

    def predict(self, candles: List[Dict]) -> Dict[str, float]:
        df = pd.DataFrame(candles)
        required_cols = {"time", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Candles must include columns {required_cols}")
        df["time"] = pd.to_datetime(df["time"])
        features_df = self.feature_engine.transform(df)
        seq_len = self.cfg.data.sequence_length
        if len(features_df) < seq_len:
            raise ValueError(f"Need at least {seq_len} candles for inference")

        features_seq = features_df.iloc[-seq_len:].values
        features_norm = (features_seq - self.scaler["mean"]) / (self.scaler["std"] + 1e-9)
        inputs = torch.from_numpy(features_norm).unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(inputs)
            buy_prob = torch.sigmoid(outputs["buy"]).item()
            sell_prob = torch.sigmoid(outputs["sell"]).item()
            direction_probs = torch.softmax(outputs["direction"], dim=-1).cpu().numpy().flatten()
            regime_probs = torch.softmax(outputs["regime"], dim=-1).cpu().numpy().flatten()

        regime_idx = int(regime_probs.argmax())
        confidence = float(np.mean([buy_prob, 1 - sell_prob, regime_probs.max()]))

        return {
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction_probs": direction_probs,
            "regime_probs": regime_probs,
            "regime_id": regime_idx,
            "regime_label": REGIME_MAP.get(regime_idx, "UNKNOWN"),
            "confidence": confidence,
        }
