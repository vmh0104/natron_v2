from __future__ import annotations

import asyncio
import json
from pathlib import Path
import numpy as np
import torch

from natron.api.server import _CONFIG, preprocess_candles
from natron.models.transformer import NatronTransformer
from natron.utils.logging import configure_logging
from natron.utils.torch_utils import get_device, load_checkpoint


class NatronSocketServer:
    def __init__(self, host: str, port: int, model_path: Path, model_cfg: dict) -> None:
        self.host = host
        self.port = port
        self.model_path = model_path
        self.model_cfg = model_cfg
        self.device = get_device()
        self.model = self._load_model()
        configure_logging()

    def _load_model(self) -> NatronTransformer:
        checkpoint, _ = load_checkpoint(self.model_path, map_location=self.device)
        model = NatronTransformer(input_dim=100, model_cfg=self.model_cfg)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        return model

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            data = await reader.readline()
            if not data:
                break
            try:
                payload = json.loads(data.decode())
                candles = payload["candles"]
                inputs = preprocess_candles(candles)
                with torch.no_grad():
                    outputs, _ = self.model(inputs)
                    buy_prob = torch.sigmoid(outputs["buy"])[0].item()
                    sell_prob = torch.sigmoid(outputs["sell"])[0].item()
                    direction_probs = torch.softmax(outputs["direction"], dim=-1)[0].cpu().numpy()
                    regime_probs = torch.softmax(outputs["regime"], dim=-1)[0].cpu().numpy()

                response = json.dumps(
                    {
                        "buy_prob": buy_prob,
                        "sell_prob": sell_prob,
                        "direction_up": float(direction_probs[1]),
                        "regime_id": int(np.argmax(regime_probs)),
                        "confidence": float(np.max(regime_probs)),
                    }
                )
                writer.write((response + "\n").encode())
                await writer.drain()
            except Exception as exc:  # noqa: BLE001
                error = json.dumps({"error": str(exc)})
                writer.write((error + "\n").encode())
                await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self) -> None:
        server = await asyncio.start_server(self.handle_client, host=self.host, port=self.port)
        async with server:
            await server.serve_forever()


def main(host: str, port: int, model_path: Path, model_cfg: dict) -> None:
    server = NatronSocketServer(host, port, model_path, model_cfg)
    asyncio.run(server.run())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Natron socket server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", type=str, default=None, help="Path to supervised model checkpoint")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else Path(_CONFIG.get("paths")["supervised_model"])
    main(args.host, args.port, model_path, _CONFIG.get("model"))
