from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from ..config import load_default_config, load_config, merge_dict
from ..data.dataset import DataModuleConfig, NatronDataModule
from ..data.feature_engine import FeatureEngineConfig
from ..data.labeling import LabelConfig
from ..data.sequence import SequenceCreatorConfig
from ..models import NatronTransformer, TransformerConfig
from ..rl.ppo import NatronRLTrainer, PPOConfig
from ..training.pretrain import NatronPretrainModule, PretrainConfig
from ..training.supervised import NatronSupervisedModule, SupervisedConfig
from ..utils.logging_utils import setup_logging
from ..utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron Transformer training pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config override")
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment name override")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to pretrained checkpoint to resume from")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    base_cfg = load_default_config()
    if args.config:
        override_cfg = load_config(args.config)
        base_cfg = merge_dict(base_cfg, override_cfg)
    if args.experiment_name:
        base_cfg["experiment"]["name"] = args.experiment_name
    return base_cfg


def prepare_datamodule(cfg: Dict[str, Any]) -> tuple[NatronDataModule, int]:
    feature_cfg = FeatureEngineConfig(**cfg.get("features", {}))
    label_cfg = LabelConfig(**cfg.get("labels", {}))
    sequence_cfg = SequenceCreatorConfig(sequence_length=cfg["data"]["sequence_length"])

    data_cfg = DataModuleConfig(
        csv_path=cfg["data"]["csv_path"],
        sequence_length=cfg["data"]["sequence_length"],
        batch_size=cfg["data"].get("batch_size", 64),
        num_workers=cfg["data"].get("num_workers", 4),
        val_split=cfg["data"].get("val_split", 0.1),
        test_split=cfg["data"].get("test_split", 0.1),
        cache_features=cfg["data"].get("feature_cache_path"),
        cache_labels=cfg["data"].get("label_cache_path"),
        feature_config=feature_cfg,
        label_config=label_cfg,
        sequence_config=sequence_cfg,
    )
    datamodule = NatronDataModule(data_cfg)
    datamodule.setup("fit")
    feature_dim = datamodule.train_dataset.sequences.size(-1)  # type: ignore[union-attr]
    return datamodule, feature_dim


def build_model(cfg: Dict[str, Any], input_dim: int) -> NatronTransformer:
    model_cfg = dict(cfg.get("model", {}))
    transformer_cfg = TransformerConfig(input_dim=input_dim, **model_cfg)
    model = NatronTransformer(transformer_cfg)
    return model


def build_trainers(cfg: Dict[str, Any]) -> tuple[pl.Trainer, pl.Trainer, Path]:
    exp_cfg = cfg.get("experiment", {})
    precision = exp_cfg.get("precision", "32-true")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator == "cpu" and precision not in {"32", "32-true"}:
        precision = "32-true"
    devices = 1

    run_dir = Path(exp_cfg.get("output_dir", "runs")) / exp_cfg.get("name", "natron")
    run_dir.mkdir(parents=True, exist_ok=True)

    pretrain_callbacks = [
        ModelCheckpoint(dirpath=run_dir / "pretrain", monitor="val/loss", save_top_k=1, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    supervised_callbacks = [
        ModelCheckpoint(dirpath=run_dir / "supervised", monitor="val/loss", save_top_k=1, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    pretrain_trainer = pl.Trainer(
        max_epochs=cfg.get("pretraining", {}).get("epochs", 10),
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=pretrain_callbacks,
        gradient_clip_val=None,
        log_every_n_steps=10,
    )

    supervised_trainer = pl.Trainer(
        max_epochs=cfg.get("supervised", {}).get("epochs", 20),
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=supervised_callbacks,
        gradient_clip_val=None,
        log_every_n_steps=10,
    )

    return pretrain_trainer, supervised_trainer, run_dir


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    setup_logging()
    set_seed(cfg.get("experiment", {}).get("seed", 42))

    datamodule, input_dim = prepare_datamodule(cfg)
    model = build_model(cfg, input_dim)

    pre_cfg_dict = cfg.get("pretraining", {}).copy()
    pre_cfg = PretrainConfig(**{k: v for k, v in pre_cfg_dict.items() if k in PretrainConfig.__annotations__})
    sup_cfg_dict = cfg.get("supervised", {}).copy()
    sup_cfg = SupervisedConfig(**{k: v for k, v in sup_cfg_dict.items() if k in SupervisedConfig.__annotations__})

    pretrain_module = NatronPretrainModule(model, pre_cfg)
    supervised_module = NatronSupervisedModule(model, sup_cfg)

    pretrain_trainer, supervised_trainer, run_dir = build_trainers(cfg)

    if args.resume_from:
        pretrain_trainer.fit(pretrain_module, datamodule=datamodule, ckpt_path=args.resume_from)
    else:
        pretrain_trainer.fit(pretrain_module, datamodule=datamodule)

    supervised_trainer.fit(supervised_module, datamodule=datamodule)
    supervised_trainer.test(supervised_module, datamodule=datamodule)

    rl_cfg_dict = cfg.get("rl", {})
    if rl_cfg_dict.get("enabled", False):
        rl_kwargs = {k: v for k, v in rl_cfg_dict.items() if k in PPOConfig.__annotations__}
        reward_cfg = rl_cfg_dict.get("reward", {})
        if isinstance(reward_cfg, dict):
            if "turnover_penalty" in reward_cfg:
                rl_kwargs["turnover_penalty"] = reward_cfg["turnover_penalty"]
            if "drawdown_penalty" in reward_cfg:
                rl_kwargs["drawdown_penalty"] = reward_cfg["drawdown_penalty"]
        rl_cfg = PPOConfig(**rl_kwargs)
        rl_trainer = NatronRLTrainer(model, datamodule, rl_cfg)
        policy_path = rl_trainer.train(run_dir / "rl")
        print(f"✅ Saved PPO policy to {policy_path}")

    model.cpu()
    model_path = Path(cfg.get("serving", {}).get("model_path", "model/natron_v2.pt"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved finetuned model to {model_path}")


if __name__ == "__main__":
    main()
