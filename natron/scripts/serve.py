from __future__ import annotations

import argparse
import threading
from pathlib import Path

from ..config import DEFAULT_CONFIG_PATH
from ..serving.api import run_app
from ..serving.service import InferenceConfig, NatronInferenceService
from ..serving.socket_server import NatronSocketServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron serving utilities")
    parser.add_argument("--mode", choices=["api", "socket", "both"], default="api")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--socket-port", type=int, default=8765)
    parser.add_argument("--model-path", default="model/natron_v2.pt")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def run_socket(host: str, port: int, model_path: str, config_path: str) -> None:
    service = NatronInferenceService(
        InferenceConfig(model_path=Path(model_path), config_path=Path(config_path))
    )
    server = NatronSocketServer(host, port, service)
    server_loop = threading.Thread(target=lambda: asyncio_run(server.run()), daemon=True)
    server_loop.start()
    server_loop.join()


def asyncio_run(coro):
    import asyncio

    asyncio.run(coro)


def main() -> None:
    args = parse_args()

    if args.mode in {"api", "both"}:
        if args.mode == "api":
            run_app(args.model_path, args.config_path, args.host, args.port)
            return
        api_thread = threading.Thread(
            target=run_app,
            args=(args.model_path, args.config_path, args.host, args.port),
            daemon=True,
        )
        api_thread.start()

    if args.mode in {"socket", "both"}:
        run_socket(args.host, args.socket_port, args.model_path, args.config_path)

    if args.mode == "both":
        api_thread.join()


if __name__ == "__main__":
    main()
