from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from .service import InferenceConfig, NatronInferenceService


class NatronSocketServer:
    def __init__(self, host: str, port: int, inference_service: NatronInferenceService) -> None:
        self.host = host
        self.port = port
        self.service = inference_service

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                try:
                    payload: Dict[str, Any] = json.loads(data.decode("utf-8"))
                    candles = payload.get("candles")
                    if not candles:
                        response = {"error": "Missing 'candles' field"}
                    else:
                        result = self.service.predict_from_candles(candles)
                        response = result
                except Exception as exc:  # pragma: no cover - runtime path
                    response = {"error": str(exc)}
                writer.write((json.dumps(response) + "\n").encode("utf-8"))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            if addr:
                print(f"Connection closed: {addr}")

    async def run(self) -> None:
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print(f"Natron socket server listening on {addrs}")
        async with server:
            await server.serve_forever()


def run_socket_server(host: str, port: int, model_path: str, config_path: str) -> None:
    service = NatronInferenceService(InferenceConfig(model_path=Path(model_path), config_path=Path(config_path)))
    server = NatronSocketServer(host, port, service)
    asyncio.run(server.run())
