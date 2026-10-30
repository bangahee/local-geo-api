"""Run real GeoCLIP inference through the API with outbound sockets blocked.

Requires requirements-dev.txt and previously downloaded model files.
Usage: OFFLINE=1 python smoke_test.py [path/to/photo.jpg]
"""
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, nargs="?")
    args = parser.parse_args()
    os.environ["OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.image is None:
        package = importlib.util.find_spec("geoclip")
        image = Path(package.origin).parent / "images" / "Kauai.png"
    else:
        image = args.image
    payload = {"image": base64.b64encode(image.read_bytes()).decode(), "top_k": 5}
    attempts = []

    def block_connection(sock, address):
        attempts.append(str(address))
        raise RuntimeError("Outbound networking disabled for offline validation")

    started = time.perf_counter()
    with patch.object(socket.socket, "connect", block_connection):
        from fastapi.testclient import TestClient
        from geolocal_api.app import create_app

        with TestClient(create_app(api_key="")) as client:
            assert client.get("/health").json()["model_state"] == "unloaded"
            response = client.post("/", json=payload)
            assert response.status_code == 200, response.text
            result = response.json()
            assert len(result["geo_predictions"]) == 5
            assert client.get("/health").json()["model_state"] == "ready"
    assert not attempts, attempts
    print(json.dumps({
        "source_image": image.name,
        "offline": True,
        "outbound_connection_attempts": len(attempts),
        "load_and_inference_seconds": round(time.perf_counter() - started, 2),
        "response": result,
    }, indent=2))


if __name__ == "__main__":
    main()
