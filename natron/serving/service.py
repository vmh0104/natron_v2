from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from ..config import DEFAULT_CONFIG_PATH, load_config, merge_dict
from ..data.feature_engine import FeatureEngine, FeatureEngineConfig
from ..models import NatronTransformer, TransformerConfig
from ..utils.logging_utils import get_logger


logger = get_logger(__name__)


REGIME_MAP = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE",
}


@dataclass(slots=True)
class InferenceConfig:
    model_path: Path
    config_path: Path = DEFAULT_CONFIG_PATH
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NatronInferenceService:
    def __init__(self, config: InferenceConfig, overrides: Dict[str, Any] | None = None) -> None:
        self.cfg = config
        base_cfg = load_config(self.cfg.config_path)
        if overrides:
            base_cfg = merge_dict(base_cfg, overrides)
        self.raw_config = base_cfg
        self.sequence_length = base_cfg["data"]["sequence_length"]

        feature_cfg = FeatureEngineConfig(**base_cfg.get("features", {}))
        self.feature_engine = FeatureEngine(feature_cfg)

        model_cfg_dict = dict(base_cfg.get("model", {}))
        self.device = torch.device(self.cfg.device)
        self.model: NatronTransformer | None = None
        self.model_cfg_dict = model_cfg_dict
        self._load_model()

    def _load_model(self) -> None:
        checkpoint_path = self.cfg.model_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        if self.model is None:
            # Temporary feature extraction to determine input dim
            dummy_df = pd.DataFrame(
                {
                    "time": pd.date_range("2020-01-01", periods=self.sequence_length, freq="15min"),
                    "open": np.ones(self.sequence_length),
                    "high": np.ones(self.sequence_length),
                    "low": np.ones(self.sequence_length),
                    "close": np.ones(self.sequence_length),
                    "volume": np.ones(self.sequence_length),
                }
            )
            dummy_features = self.feature_engine.transform(dummy_df)
            input_dim = dummy_features.shape[1]
            model_cfg = dict(self.model_cfg_dict)
            model_cfg.pop("input_dim", None)
            transformer_cfg = TransformerConfig(input_dim=input_dim, **model_cfg)
            self.model = NatronTransformer(transformer_cfg)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded model from %s", checkpoint_path)

    def predict_from_candles(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = pd.DataFrame(candles)
        if len(df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} candles, received {len(df)}")
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        features = self.feature_engine.transform(df)
        feature_tensor = torch.from_numpy(features.to_numpy(dtype=np.float32)[-self.sequence_length :]).unsqueeze(0)
        feature_tensor = feature_tensor.to(self.device)

        if self.model is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            outputs = self.model(feature_tensor)
        buy_prob = torch.sigmoid(outputs.buy_logits).item()
        sell_prob = torch.sigmoid(outputs.sell_logits).item()
        direction_probs = torch.softmax(outputs.direction_logits, dim=-1).cpu().numpy().tolist()
        regime_probs = torch.softmax(outputs.regime_logits, dim=-1).cpu().numpy().tolist()

        regime_id = int(np.argmax(regime_probs))
        confidence = float(max(buy_prob, sell_prob, max(direction_probs)))
        response = {
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction_up": float(direction_probs[1]) if len(direction_probs) > 1 else float(direction_probs[0]),
            "direction_probs": direction_probs,
            "regime": REGIME_MAP.get(regime_id, str(regime_id)),
            "regime_probs": regime_probs,
            "confidence": confidence,
        }
        return response
