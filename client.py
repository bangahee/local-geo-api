"""Example: python client.py /path/to/photo.jpg --top-k 5"""
import argparse
import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765/")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    payload = {"image": base64.b64encode(args.image.read_bytes()).decode("ascii"),
               "top_k": args.top_k}
    headers = {"Content-Type": "application/json"}
    if os.getenv("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]
    request = Request(args.url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urlopen(request, timeout=600) as response:
            print(json.dumps(json.load(response), indent=2))
    except HTTPError as exc:
        print(exc.read().decode())
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
