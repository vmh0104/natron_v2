"""Inference utilities for Natron runtime prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from natron.data.feature_engineer import FeatureEngine
from natron.models.transformer import NatronModelConfig, NatronTransformer
from natron.training.reinforcement import NatronRLPolicy

REGIME_NAMES = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE",
}

DIRECTION_NAMES = {0: "DOWN", 1: "UP", 2: "NEUTRAL"}


@dataclass
class NatronModelBundle:
    model: NatronTransformer
    rl_policy: Optional[NatronRLPolicy]
    feature_columns: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    window_length: int
    device: torch.device

    @classmethod
    def load(cls, checkpoint_path: Path, device: Optional[str] = None) -> "NatronModelBundle":
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_cfg = NatronModelConfig(**checkpoint["model_config"])
        model = NatronTransformer(model_cfg)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        rl_state = checkpoint.get("rl_policy_state")
        rl_policy = None
        if rl_state:
            rl_policy = NatronRLPolicy(model.get_encoder(), train_encoder=False)
            rl_policy.load_state_dict(rl_state)
            rl_policy.to(device)
            rl_policy.eval()

        feature_columns = checkpoint["feature_columns"]
        scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
        scaler_scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)
        window_length = checkpoint.get("window_length", 96)

        return cls(
            model=model,
            rl_policy=rl_policy,
            feature_columns=feature_columns,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            window_length=window_length,
            device=device,
        )

    def scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        aligned = features[self.feature_columns]
        scaled_values = (aligned.values - self.scaler_mean) / np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
        return pd.DataFrame(scaled_values, index=aligned.index, columns=aligned.columns)


class NatronPredictor:
    """High-level helper that turns raw OHLCV candles into predictions."""

    def __init__(self, bundle: NatronModelBundle) -> None:
        self.bundle = bundle
        self.feature_engine = FeatureEngine()

    def predict(self, candles: pd.DataFrame | Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        df = self._ensure_dataframe(candles)
        if len(df) < self.bundle.window_length:
            raise ValueError(
                f"Received {len(df)} candles but need at least {self.bundle.window_length} for inference."
            )

        features = self.feature_engine.transform(df)
        scaled = self.bundle.scale_features(features)
        sequence = scaled.tail(self.bundle.window_length).to_numpy(dtype=np.float32)
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.bundle.device)

        with torch.no_grad():
            buy_logits, sell_logits, direction_logits, regime_logits = self.bundle.model(sequence_tensor)
            buy_prob = torch.sigmoid(buy_logits.squeeze(-1)).item()
            sell_prob = torch.sigmoid(sell_logits.squeeze(-1)).item()
            direction_probs = torch.softmax(direction_logits, dim=-1).squeeze(0).cpu().numpy()
            regime_probs = torch.softmax(regime_logits, dim=-1).squeeze(0).cpu().numpy()

        direction_idx = int(direction_probs.argmax())
        regime_idx = int(regime_probs.argmax())
        confidence = float(np.max([direction_probs.max(), regime_probs.max(), buy_prob, sell_prob]))

        response: Dict[str, Any] = {
            "buy_prob": round(buy_prob, 4),
            "sell_prob": round(sell_prob, 4),
            "direction": DIRECTION_NAMES.get(direction_idx, "UNKNOWN"),
            "direction_probs": {
                DIRECTION_NAMES[i]: round(float(direction_probs[i]), 4) for i in range(len(direction_probs))
            },
            "regime": REGIME_NAMES.get(regime_idx, "UNKNOWN"),
            "regime_probs": {
                REGIME_NAMES[i]: round(float(regime_probs[i]), 4) for i in range(len(regime_probs))
            },
            "confidence": round(confidence, 4),
            "timestamp": self._latest_timestamp(df),
        }

        if self.bundle.rl_policy is not None:
            with torch.no_grad():
                dist, _, _ = self.bundle.rl_policy(sequence_tensor)
                action_probs = dist.probs.squeeze(0).cpu().numpy()
            action_idx = int(action_probs.argmax())
            response.update(
                {
                    "rl_action": self._map_action(action_idx),
                    "rl_action_probs": {
                        "LONG": round(float(action_probs[0]), 4),
                        "FLAT": round(float(action_probs[1]), 4),
                        "SHORT": round(float(action_probs[2]), 4),
                    },
                }
            )

        return response

    @staticmethod
    def _map_action(action_idx: int) -> str:
        return {0: "LONG", 1: "FLAT", 2: "SHORT"}.get(action_idx, "UNKNOWN")

    @staticmethod
    def _ensure_dataframe(data: pd.DataFrame | Iterable[Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(list(data))
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time")
        return df

    @staticmethod
    def _latest_timestamp(df: pd.DataFrame) -> Optional[str]:
        if "time" in df.columns:
            ts = df["time"].iloc[-1]
            if isinstance(ts, pd.Timestamp):
                return ts.isoformat()
            return str(ts)
        return None


def load_predictor(model_path: Path, device: Optional[str] = None) -> NatronPredictor:
    bundle = NatronModelBundle.load(model_path, device)
    return NatronPredictor(bundle)


__all__ = ["NatronModelBundle", "NatronPredictor", "load_predictor"]
