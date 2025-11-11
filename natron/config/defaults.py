from pathlib import Path
from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    model_root = project_root / "model"

    return {
        "paths": {
            "project_root": str(project_root),
            "data_root": str(data_root),
            "raw_data": str(data_root / "data_export.csv"),
            "artifacts": str(project_root / "artifacts"),
            "model_dir": str(model_root),
            "pretrained_encoder": str(model_root / "natron_encoder.pt"),
            "supervised_model": str(model_root / "natron_v2.pt"),
        },
        "data": {
            "sequence_length": 96,
            "features": {
                "rolling_windows": [5, 10, 14, 20, 26, 50, 96],
                "neutral_buffer": 0.001,
            },
            "train": {
                "validation_split": 0.2,
                "test_split": 0.1,
                "shuffle": True,
                "random_seed": 42,
            },
        },
        "pretraining": {
            "epochs": 50,
            "batch_size": 64,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "mask_prob": 0.15,
            "temperature": 0.1,
            "use_amp": True,
        },
        "supervised": {
            "epochs": 100,
            "batch_size": 64,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "scheduler": {
                "mode": "min",
                "factor": 0.5,
                "patience": 5,
                "threshold": 1e-4,
            },
            "use_amp": True,
            "class_weights": {
                "buy": 1.0,
                "sell": 1.0,
                "direction": [1.0, 1.0, 1.0],
                "regime": [1.0] * 6,
            },
        },
        "rl": {
            "algorithm": "ppo",
            "total_steps": 200_000,
            "update_interval": 2_000,
            "reward": {
                "alpha": 0.001,
                "beta": 0.001,
            },
        },
        "model": {
            "d_model": 192,
            "n_heads": 8,
            "num_layers": 6,
            "d_ff": 512,
            "dropout": 0.1,
            "max_len": 512,
            "embedding": {
                "type": "learned",
                "dropout": 0.1,
            },
        },
        "hardware": {
            "device": "cuda",
            "num_workers": 8,
            "pin_memory": True,
            "seed": 1337,
        },
    }
