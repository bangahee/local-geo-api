import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import threading

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from geolocal_api.app import MAX_REQUEST_BYTES, create_app
from geolocal_api.images import InvalidImage, prepare_image


def encoded_image():
    buffer = BytesIO()
    image = Image.new("RGB", (32, 24), (20, 100, 200))
    exif = Image.Exif()
    exif[270] = "test metadata that must be removed"
    image.save(buffer, format="JPEG", exif=exif)
    return base64.b64encode(buffer.getvalue()).decode()


class RecordingBackend:
    state = "ready"

    def __init__(self):
        self.paths = []

    def predict(self, path, top_k):
        self.paths.append(path)
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert not image.getexif()
            assert not image.info
        return [{"coordinates": [35.0 + i, 127.0], "score": 0.01 / (i + 1)}
                for i in range(top_k)]


@pytest.fixture
def backend():
    return RecordingBackend()


@pytest.fixture
def client(backend):
    with TestClient(create_app(backend, api_key="test-key")) as client:
        yield client


@pytest.mark.parametrize("route", ["/", "/v1/geolocate"])
def test_request_contract_and_cleanup(client, backend, route):
    response = client.post(route, json={"image": encoded_image(), "key": "test-key", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == 200
    assert body["message"] == "Success"
    assert len(body["geo_predictions"]) == 3
    assert body["geo_predictions"][0] == {"coordinates": [35.0, 127.0], "score": 0.01}
    assert not backend.paths[0].exists()
    assert not backend.paths[0].parent.exists()


def test_auth_header_and_rejection(client, backend):
    assert client.post("/", json={"image": "a"}).status_code == 401
    response = client.post("/", headers={"X-API-Key": "test-key"}, json={"image": encoded_image()})
    assert response.status_code == 200
    assert len(backend.paths) == 1


@pytest.mark.parametrize("changes", [
    {"top_k": 0}, {"top_k": 21}, {"top_k": True}, {"top_k": "5"},
    {"use_classification": True}, {"anti_cluster": True}, {"extra": "unsupported"},
])
def test_invalid_options(client, backend, changes):
    response = client.post("/", json={"image": encoded_image(), "key": "test-key", **changes})
    assert response.status_code == 422
    assert response.json()["status"] == 422
    assert not backend.paths


@pytest.mark.parametrize("image", ["%%%", "a", base64.b64encode(b"not a photo").decode()])
def test_bad_image(client, backend, image):
    response = client.post("/", json={"image": image, "key": "test-key"})
    assert response.status_code == 400
    assert not backend.paths


def test_data_url(client):
    response = client.post("/", json={
        "image": "data:image/jpeg;base64," + encoded_image(), "key": "test-key"})
    assert response.status_code == 200


def test_streamed_body_limit(client, backend):
    chunks = (b"x" * 1024 * 1024 for _ in range(MAX_REQUEST_BYTES // (1024 * 1024) + 1))
    response = client.post("/", content=chunks, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert not backend.paths


def test_errors_do_not_echo_secrets(client):
    response = client.post("/", json={"image": "private-photo", "key": "secret", "top_k": "secret"})
    assert response.status_code == 422
    assert "secret" not in response.text
    assert "private-photo" not in response.text


def test_failure_cleanup_and_lock_release():
    class FailingBackend(RecordingBackend):
        def predict(self, path, top_k):
            self.paths.append(path)
            raise RuntimeError("test failure")

    backend = FailingBackend()
    with TestClient(create_app(backend, api_key="")) as client:
        for _ in range(2):
            response = client.post("/", json={"image": encoded_image()})
            assert response.status_code == 503
            assert "test failure" not in response.text
    assert len(backend.paths) == 2
    assert all(not path.parent.exists() for path in backend.paths)


def test_busy_requests_are_rejected_and_health_remains_responsive():
    started, release = threading.Event(), threading.Event()

    class BlockingBackend(RecordingBackend):
        def predict(self, path, top_k):
            started.set()
            assert release.wait(10)
            return super().predict(path, top_k)

    with TestClient(create_app(BlockingBackend(), api_key="")) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post, "/", json={"image": encoded_image()})
            try:
                assert started.wait(5)
                assert client.get("/health").json()["model_state"] == "ready"
                response = client.post("/", json={"image": encoded_image()})
                assert response.status_code == 503
                assert response.headers["retry-after"] == "2"
            finally:
                release.set()
            assert first.result(timeout=5).status_code == 200


def test_pixel_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("geolocal_api.images.MAX_PIXELS", 10)
    with pytest.raises(InvalidImage, match="megapixel"):
        prepare_image(encoded_image(), tmp_path / "image.png")


def test_animation_rejected(tmp_path):
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(
        buffer, format="PNG", save_all=True, append_images=[Image.new("RGB", (2, 2), "blue")])
    with pytest.raises(InvalidImage, match="Animated"):
        prepare_image(base64.b64encode(buffer.getvalue()).decode(), tmp_path / "image.png")
