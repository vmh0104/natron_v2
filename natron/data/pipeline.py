from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from natron.data.feature_engine import FeatureEngine
from natron.data.labeling_v2 import LabelGeneratorV2
from natron.data.sequence import SequenceCreator


@dataclass
class DataPipeline:
    data_path: Path
    sequence_length: int = 96
    neutral_buffer: float = 0.001

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Tuple]:
        raw = self._load_data()
        features = FeatureEngine(neutral_buffer=self.neutral_buffer).transform(raw)
        labels = LabelGeneratorV2(neutral_buffer=self.neutral_buffer).generate(raw, features)
        sequences, targets = SequenceCreator(sequence_length=self.sequence_length).create(features, labels)
        return raw, features, labels, (sequences, targets)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
