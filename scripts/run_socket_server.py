#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from natron.serving.socket_server import run_socket_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Natron TCP prediction bridge")
    parser.add_argument("--model", type=Path, required=True, help="Path to natron_v2.pt checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on")
    parser.add_argument("--device", default=None, help="Optional device override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_socket_server(args.model, host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
