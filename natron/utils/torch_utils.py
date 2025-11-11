from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def autocast(enabled: bool, device: torch.device) -> Iterator[None]:
    if enabled and device.type == "cuda":
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def save_checkpoint(
    state: dict,
    path: Path,
    is_best: bool = False,
    best_suffix: str = "_best",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.with_name(path.stem + best_suffix + path.suffix)
        torch.save(state, best_path)


def load_checkpoint(path: Path, map_location: Optional[str] = None) -> Tuple[dict, Path]:
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint, path
