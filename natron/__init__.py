"""
Natron Transformer package.

Provides data processing, modeling, training, reinforcement learning, and inference
utilities for the Natron multi-task financial trading system.
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "get_version",
]


def get_version() -> str:
    """Return the package version if installed, otherwise '0.0.0'."""
    try:
        return version("natron")
    except PackageNotFoundError:
        return "0.0.0"
