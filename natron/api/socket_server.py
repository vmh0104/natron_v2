"""Socket server bridge for MQL5 EA integration."""

from __future__ import annotations

import argparse
import json
import socketserver

from natron.api.predictor import NatronPredictor
from natron.utils import create_logger


class NatronRequestHandler(socketserver.BaseRequestHandler):
    predictor: NatronPredictor = None  # type: ignore
    logger = create_logger("natron.socket")

    def handle(self) -> None:
        data = self.request.recv(65536).strip()
        if not data:
            return
        try:
            payload = json.loads(data.decode("utf-8"))
            candles = payload.get("candles", [])
            result = self.predictor.predict(candles)
            response = {
                "buy_prob": result["buy_prob"],
                "sell_prob": result["sell_prob"],
                "direction_probs": result["direction_probs"].tolist(),
                "regime_probs": result["regime_probs"].tolist(),
                "regime": result["regime_label"],
                "confidence": result["confidence"],
            }
            message = json.dumps(response).encode("utf-8")
            self.request.sendall(message)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to process request: %s", exc)
            error = json.dumps({"error": str(exc)}).encode("utf-8")
            self.request.sendall(error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron Socket Server")
    parser.add_argument("--config", type=str, default="natron/configs/natron_base.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = NatronPredictor(args.config)
    NatronRequestHandler.predictor = predictor
    logger = create_logger("natron.socket.server")
    server = socketserver.ThreadingTCPServer((args.host, args.port), NatronRequestHandler)
    logger.info("Natron socket server listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down socket server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
