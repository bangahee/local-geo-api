# Validation — October 30, 2026

Environment: macOS ARM64, Python 3.12.15, CPU, GeoCLIP 1.3.0. Complete installed
Python dependency versions are in `requirements-tested.txt`.

## API tests

`python -m pytest -q`: **20 passed**, in 6.73 seconds. One dependency deprecation
warning concerned Starlette's httpx test adapter. The tests use a substitute
backend to isolate API behavior from model downloads and model quality.

## Real model, offline

Ran `smoke_test.py` using the photo `geoclip/images/Kauai.png` bundled with the
upstream Python package. The test set offline environment variables and blocked
Python socket connection attempts. It exercised the real model through the
FastAPI endpoint, using previously cached weights.

- HTTP 200; five ranked coordinate candidates returned.
- Model state changed from `unloaded` to `ready`.
- Zero outbound socket connection attempts.
- Loading plus inference and test setup: **6.37 seconds**.
- First candidate: **22.197973, -159.621902**, in Kauai.
- Complete output: `example-result.json`.

Model revision: `40a931123d2c6451ef7da1d37cf68cded0ec4d35` from
`VicenteVivanco/geoclip-geolocation`. The backbone revision is specified by the
upstream model configuration. The same model revision is pinned by default in
this API.

Also started Uvicorn on localhost port 8765 with `OFFLINE=1` and called the real
HTTP endpoint through `client.py`, requesting three predictions. The client
received HTTP 200 and three candidates. `/health` remained responsive during
model loading and subsequently reported `ready`.

This is a functional smoke test on one upstream example, not an independent
accuracy benchmark. GeoSpy equivalence, street-level accuracy, GPU support,
Windows/Linux installation, and high-volume deployment have not been validated.
