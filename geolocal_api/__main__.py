import argparse


def main():
    parser = argparse.ArgumentParser(description="Local image geolocation")
    parser.add_argument("command", choices=["serve", "download-model"], nargs="?", default="serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    if args.command == "download-model":
        from .model import GeoCLIPBackend
        GeoCLIPBackend().load()
        print("Model and backbone cached. You can now serve with OFFLINE=1.")
    else:
        import uvicorn
        uvicorn.run("geolocal_api.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
