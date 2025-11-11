#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from natron.serving.api import run_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Natron predictions via Flask API")
    parser.add_argument("--model", type=Path, required=True, help="Path to natron_v2.pt checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--device", default=None, help="Optional device override (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_app(args.model, host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
