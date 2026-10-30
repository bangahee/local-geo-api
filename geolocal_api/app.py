import logging
import os
from pathlib import Path
import secrets
import tempfile
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from starlette.concurrency import run_in_threadpool

from .images import InvalidImage, prepare_image
from .model import GeoCLIPBackend


logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 12 * 1024 * 1024


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(min_length=1, description="Base64 JPEG, PNG, or WebP")
    key: str | None = Field(default=None, repr=False)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    use_classification: StrictBool = False
    anti_cluster: StrictBool = False


class Prediction(BaseModel):
    coordinates: tuple[float, float]
    score: float


class PredictionResponse(BaseModel):
    status: int = 200
    message: str = "Success"
    geo_predictions: list[Prediction]


class BodyLimitMiddleware:
    """Bound the received body even when Content-Length is absent or incorrect."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        parts, size = [], 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body = event.get("body", b"")
            size += len(body)
            if size > MAX_REQUEST_BYTES:
                response = JSONResponse(
                    {"status": 413, "message": "Request exceeds the 12 MiB limit"},
                    status_code=413,
                )
                return await response(scope, receive, send)
            parts.append(body)
            if not event.get("more_body", False):
                break
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(parts), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(backend=None, api_key: str | None = None):
    app = FastAPI(title="Local Geo API", version="0.1.0")
    app.add_middleware(BodyLimitMiddleware)
    engine = backend if backend is not None else GeoCLIPBackend()
    configured_key = os.getenv("API_KEY", "") if api_key is None else api_key
    inference_lock = threading.Lock()

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(
            {"status": exc.status_code, "message": exc.detail},
            status_code=exc.status_code, headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never echo base64 image content or API keys in error messages.
        fields = sorted({str(error["loc"][-1]) for error in exc.errors()})
        return JSONResponse(
            {"status": 422, "message": "Invalid request fields: " + ", ".join(fields)},
            status_code=422,
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "geoclip", "model_state": engine.state,
                "authentication_enabled": bool(configured_key)}

    def predict(payload):
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(503, "Model is busy; retry shortly", headers={"Retry-After": "2"})
        try:
            # Private temporary directory removed on success and on failures.
            with tempfile.TemporaryDirectory(prefix="local-geo-") as folder:
                image_path = Path(folder) / "image.png"
                try:
                    prepare_image(payload.image, image_path)
                except InvalidImage as exc:
                    raise HTTPException(400, str(exc)) from exc
                try:
                    predictions = engine.predict(image_path, payload.top_k)
                except Exception as exc:
                    logger.exception("Local model inference failed")
                    raise HTTPException(
                        503, "Model unavailable; check server logs and download model files"
                    ) from exc
                return {"status": 200, "message": "Success", "geo_predictions": predictions}
        finally:
            inference_lock.release()

    @app.post("/", response_model=PredictionResponse)
    @app.post("/v1/geolocate", response_model=PredictionResponse)
    async def geolocate(payload: PredictionRequest, request: Request):
        provided = request.headers.get("X-API-Key") or payload.key or ""
        if configured_key and not secrets.compare_digest(
            configured_key.encode(), provided.encode()
        ):
            raise HTTPException(401, "Invalid API key")
        if payload.use_classification or payload.anti_cluster:
            raise HTTPException(
                422, "use_classification and anti_cluster are not supported by this backend"
            )
        return await run_in_threadpool(predict, payload)

    return app


app = create_app()
