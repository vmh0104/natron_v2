#!/usr/bin/env python3
from __future__ import annotations

import argparse

from natron.api.server import create_app


def main(host: str, port: int) -> None:
    app = create_app()
    app.run(host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Natron Flask API")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(args.host, args.port)
