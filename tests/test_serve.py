"""The /match endpoint with stand-in detection and embedding: no models are loaded."""
import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from pawmatch import detect, serve  # noqa: E402

GALLERY = np.eye(6)  # six dogs, one unit vector each
DOGS = np.array([f"dog{i}" for i in range(6)])


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, "PNG")
    return buf.getvalue()


def _client(threshold=0.5):
    def detect_faces(images):  # a black photo has no face
        return [None if max(im.getextrema()[0]) == 0 else ((8.0, 4.0, 40.0, 36.0), 0.9) for im in images]

    def embed_face(face):  # red photos look like dog0, with a little of dog1
        v = np.array([0.9, 0.3, 0.1, 0, 0, 0])
        return v / np.linalg.norm(v)

    return TestClient(serve.create_app(serve.Matcher(detect_faces, embed_face, GALLERY, DOGS, threshold)))


def test_match_returns_top5_sorted_with_no_match_flag():
    r = _client().post("/match", files={"photo": ("dog.png", _png("red"))}, data={"description": "brown lab"})
    assert r.status_code == 200
    body = r.json()
    assert [m["dog_id"] for m in body["matches"]] == ["dog0", "dog1", "dog2", "dog3", "dog4"]
    assert body["matches"][0]["score"] == pytest.approx(0.9 / np.linalg.norm([0.9, 0.3, 0.1]))
    assert body["no_match"] is False and body["description_used"] is False
    assert "CC BY 4.0" in body["attribution"] and body["face"]["box"] == [8.0, 4.0, 40.0, 36.0]
    strict = _client(threshold=0.99).post("/match", files={"photo": ("dog.png", _png("red"))})
    assert strict.json()["no_match"] is True


def test_no_face_is_a_clear_error():
    r = _client().post("/match", files={"photo": ("dark.png", _png("black"))})
    assert r.status_code == 422 and r.json()["detail"] == serve.NO_FACE


def test_non_image_is_rejected():
    r = _client().post("/match", files={"photo": ("notes.txt", b"not an image")})
    assert r.status_code == 400 and r.json()["detail"] == serve.NOT_IMAGE


def test_align_is_a_square_crop_around_the_box():
    img = Image.new("RGB", (100, 80), "white")
    out = detect.align(img, (20, 10, 60, 40))  # 40 x 30 box -> 48 px square, resized to 224
    assert out.size == (224, 224)
