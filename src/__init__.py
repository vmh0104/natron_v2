"""
Natron Transformer - Multi-Task Financial Trading Model
End-to-End GPU Pipeline for Deep Learning-Based Trading
"""

__version__ = "2.0.0"
__author__ = "Natron AI Team"

from .feature_engine import FeatureEngine
from .label_generator import LabelGeneratorV2
from .sequence_creator import SequenceCreator
from .model import NatronTransformer

__all__ = [
    "FeatureEngine",
    "LabelGeneratorV2",
    "SequenceCreator",
    "NatronTransformer",
]
