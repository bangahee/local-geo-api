# Local Geo API

Self-hosted image-to-location predictions using the open GeoCLIP model. No paid
API key or remote inference service is required. This is an independent
implementation, not GeoSpy's model or an official GeoSpy service.

## Run locally

Use Python 3.12 (recommended) or 3.11–3.13. From this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m geolocal_api download-model
OFFLINE=1 python -m geolocal_api serve
```

Installation and the first model download need internet access. Allow a few GB
for the Python environment, weights, and caches. The first model load can take
several minutes. Once cached, `OFFLINE=1` loads only local model files. Photos
are processed on your machine; the model does not call an inference provider.
CPU is the default. Optional `DEVICE=cuda` or `DEVICE=mps` requires a compatible
PyTorch installation and device; those device configurations need separate testing.

Open **http://127.0.0.1:8765/docs** for the interactive API documentation.
The server binds to localhost by default and uses one worker, keeping one model
in memory. `/health` reports `unloaded`, `loading`, `ready`, or `failed`; HTTP
200 on this endpoint describes the server, not model readiness. The model loads
on the first prediction unless separately downloaded and cached beforehand.

In another terminal with the environment activated:

```sh
python client.py /absolute/path/to/photo.jpg --top-k 5
```

## API

`POST /` or `POST /v1/geolocate`, JSON body:

```json
{
  "image": "BASE64_IMAGE_BYTES",
  "top_k": 5,
  "use_classification": false,
  "anti_cluster": false
}
```

`image` also accepts a JPEG, PNG, or WebP base64 data URL. `top_k` is an integer
from 1 to 20, default 5. Example response structure (illustrative coordinates):

```json
{
  "status": 200,
  "message": "Success",
  "geo_predictions": [
    {"coordinates": [48.8584, 2.2945], "score": 0.018}
  ]
}
```

Coordinates are `[latitude, longitude]`. Scores rank candidates within the
model's location gallery. They are **not calibrated probabilities of a correct
location**, and are not comparable to GeoSpy scores. Decimal precision does not
establish street-level accuracy. Visually ambiguous photos can be wrong by
hundreds or thousands of kilometers. This implementation has not been benchmarked
against GeoSpy.

The basic GeoSpy-style `image`, `key`, `top_k` request and
`status`, `message`, `geo_predictions` response are supported. This is not full
behavioral compatibility: `use_classification=true` and `anti_cluster=true`
return HTTP 422; `similarity_score_1km` is not provided. GeoCLIP cannot recreate
the original proprietary model or its geographic coverage.

## Local access and storage

Authentication is optional for localhost. To require your own access token:

```sh
API_KEY='choose-your-own-token' OFFLINE=1 python -m geolocal_api serve
```

Clients can send `X-API-Key` or the legacy JSON `key` field. The example client
reads `API_KEY` from its environment. This token is created by you, costs nothing,
and is unrelated to model access. If exposing the API beyond localhost, configure
authentication and place it behind a TLS reverse proxy with connection limits.

Images are limited to 8 MiB decoded, requests to 12 MiB, and images to 25 million
pixels. Animated files are rejected. EXIF orientation is applied, metadata is
removed, and pixel dimensions are reduced to at most 2048 before inference.
Normalized images exist in private temporary directories during a request and
are removed on completion or handled errors. An abrupt process termination can
leave temporary files in the operating system's temporary directory. Request
content is not logged by this app. Public API documentation uses browser resources
from a CDN; it is not needed for predictions or the CLI client to work offline.

One prediction runs at a time. Concurrent prediction attempts return HTTP 503
with `Retry-After: 2`. Malformed images return 400, authentication failures 401,
oversized requests 413, invalid fields/options 422, and unavailable model files
or inference failures 503. Model loading failures can be retried after fixing
the cause. This is a local prototype, without multi-user rate limiting or a queue.

Optional environment variables:

| Variable | Meaning |
| --- | --- |
| `API_KEY` | Your optional local access token |
| `OFFLINE=1` | Require cached model and backbone files |
| `DEVICE` | PyTorch device, default `cpu` |
| `HF_HOME` | Optional writable model cache directory; keep the same value for download and serving |
| `MODEL_REVISION` | Override the default pinned model commit; use the same value for download and serving |

## Verify

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Contract tests use a small substitute backend to isolate request handling from
model quality. They cover authentication, bounds, image validation, metadata
removal, temporary-file cleanup, model failures, and concurrent requests. Real
model validation is recorded separately in `VALIDATION.md`.
`requirements-tested.txt` records the complete Python dependency versions used
for validation on macOS ARM64. It includes development dependencies.

## Sources and attribution

- [GeoCLIP official implementation](https://github.com/VicenteVivan/geo-clip)
- [GeoCLIP model and limitations](https://huggingface.co/VicenteVivanco/geoclip-geolocation)
- [GeoCLIP paper](https://arxiv.org/abs/2309.16020)
- [Legacy GeoSpy API schema](https://dev.geospy.ai/docs)
- [Official GeoSpy → Raven transition](https://graylark.com/geospy)

GeoCLIP is distributed by its authors under the MIT license; its dependencies
retain their respective licenses. Model files are fetched from the authors'
repository and are not bundled in this project.
