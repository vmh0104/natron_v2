"""Async socket server bridging Natron with external trading clients."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from natron.pipelines.inference import load_predictor
from natron.utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


class NatronSocketServer:
    def __init__(self, model_path: Path, host: str, port: int, device: Optional[str] = None) -> None:
        self.predictor = load_predictor(model_path, device)
        self.host = host
        self.port = port

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.info("Client connected: %s", peer)
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                message = data.decode().strip()
                if not message:
                    continue
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    await self._write_response(writer, {"error": "invalid_json"})
                    continue

                response = await self._process_payload(payload)
                await self._write_response(writer, response)
        except asyncio.CancelledError:  # pragma: no cover - server shutdown
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error handling client %s", peer)
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info("Client disconnected: %s", peer)

    async def _process_payload(self, payload: dict) -> dict:
        msg_type = payload.get("type")
        if msg_type == "ping":
            return {"type": "pong"}
        if msg_type == "predict":
            candles = payload.get("candles", [])
            try:
                prediction = self.predictor.predict(candles)
                return {"type": "prediction", "data": prediction}
            except ValueError as exc:
                return {"type": "error", "message": str(exc)}
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error during prediction")
                return {"type": "error", "message": "internal_error"}
        return {"type": "error", "message": "unknown_type"}

    async def _write_response(self, writer: asyncio.StreamWriter, response: dict) -> None:
        writer.write((json.dumps(response) + "\n").encode())
        await writer.drain()

    async def start(self) -> None:
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        logger.info("Natron socket server listening on %s", addresses)
        async with server:
            await server.serve_forever()


def run_socket_server(model_path: Path, host: str = "0.0.0.0", port: int = 9000, device: Optional[str] = None) -> None:
    setup_logging(Path("logs"), "natron_socket")
    server = NatronSocketServer(model_path, host, port, device)
    asyncio.run(server.start())


__all__ = ["NatronSocketServer", "run_socket_server"]
