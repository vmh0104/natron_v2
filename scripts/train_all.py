#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(path)] + args
    subprocess.check_call(cmd)


def main(config: str | None) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_args = ["--config", config] if config else []

    run_script(repo_root / "scripts" / "prepare_data.py", script_args)
    run_script(repo_root / "scripts" / "train_pretrain.py", script_args)
    run_script(repo_root / "scripts" / "train_supervised.py", script_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full Natron training pipeline")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
