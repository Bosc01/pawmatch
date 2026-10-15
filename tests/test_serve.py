"""The /match endpoint with stand-in detection and embedding: no models are loaded."""
import io
import json
from pathlib import Path

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
    impostors = np.full(1000, threshold - 1e-3)  # every impostor score just under the intended threshold
    def detect_faces(images):  # a black photo has no face
        return [None if max(im.getextrema()[0]) == 0 else ((8.0, 4.0, 40.0, 36.0), 0.9) for im in images]

    def embed_face(face):  # red photos look like dog0, with a little of dog1
        v = np.array([0.9, 0.3, 0.1, 0, 0, 0])
        return v / np.linalg.norm(v)

    return TestClient(serve.create_app(serve.Matcher(detect_faces, embed_face, GALLERY, DOGS, impostors, 0.1)))


def test_match_returns_top5_sorted_with_a_confidence_label():
    r = _client().post("/match", files={"photo": ("dog.png", _png("red"))}, data={"description": "brown lab"})
    assert r.status_code == 200
    body = r.json()
    assert [m["dog_id"] for m in body["matches"]] == ["dog0", "dog1", "dog2", "dog3", "dog4"]
    assert body["matches"][0]["score"] == pytest.approx(0.9 / np.linalg.norm([0.9, 0.3, 0.1]))
    assert body["confidence"] == "confident_match" and body["description_used"] is False
    assert body["confidence_calibration"] == serve.CONFIDENCE_CALIBRATION
    assert body["operating_point"] == {"gallery_far": 0.1, "gallery_size": 6}
    assert "CC BY 4.0" in body["attribution"] and body["face"]["box"] == [8.0, 4.0, 40.0, 36.0]
    assert all(set(m) == {"rank", "dog_id", "score"} for m in body["matches"])  # no gallery photos were given
    assert body["face"]["route"] == "face_box" and body["face"]["note"] is None
    assert body["detector_attribution"] == serve.DETECTOR_ATTRIBUTION[detect.DETECTOR]
    assert "Oxford-IIIT Pet" in serve.DETECTOR_ATTRIBUTION["frcnn_head"] and "CC BY-SA 4.0" in serve.DETECTOR_ATTRIBUTION["frcnn_head"]
    strict = _client(threshold=0.99).post("/match", files={"photo": ("dog.png", _png("red"))}).json()
    assert strict["confidence"] == "uncertain" and len(strict["matches"]) == 5  # a label, not a filter


def test_no_face_falls_back_to_the_whole_photo_and_says_so():
    r = _client().post("/match", files={"photo": ("dark.png", _png("black"))})
    face = r.json()["face"]
    assert r.status_code == 200 and face["route"] == "whole_photo_no_box" and face["box"] is None
    assert face["note"] == serve.WHOLE_PHOTO["whole_photo_no_box"]


def test_face_crop_routes():
    img = Image.new("RGB", (100, 80), "white")
    assert detect.face_crop(img, None)[1] == "whole_photo_no_box"
    assert detect.face_crop(img, ((0, 0, 90, 70), 0.5))[1] == "whole_photo_box_covers_most"  # 79% of the photo
    crop, route = detect.face_crop(img, ((20, 10, 60, 40), 0.5))  # 15% of the photo
    assert route == "face_box" and crop.size == (224, 224)


def test_non_image_is_rejected():
    r = _client().post("/match", files={"photo": ("notes.txt", b"not an image")})
    assert r.status_code == 400 and r.json()["detail"] == serve.NOT_IMAGE


def test_align_is_a_square_crop_around_the_box():
    img = Image.new("RGB", (100, 80), "white")
    out = detect.align(img, (20, 10, 60, 40))  # 40 x 30 box -> 48 px square, resized to 224
    assert out.size == (224, 224)


def test_iou_hand_computed():
    assert detect.iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert detect.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert detect.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_align_stays_inside_the_photo():
    img = Image.new("RGB", (100, 80), "white")
    out = detect.align(img, (70, 50, 98, 78))  # box at the corner: the square shifts inward, no black fill
    assert out.getextrema() == ((255, 255), (255, 255), (255, 255))
    out = detect.align(img, (0, 0, 100, 80))  # 1.2 x 100 > 80: the side is capped at the shorter edge
    assert out.size == (224, 224) and out.getextrema() == ((255, 255), (255, 255), (255, 255))


def test_service_detector_matches_the_latest_half_a_choice():
    trained, zero_shot = Path("results/detection_oxford_trained.json"), Path("results/detection_oxford.json")
    if trained.exists():
        chosen = json.loads(trained.read_text())["internal_val_service_choice"]
    elif zero_shot.exists():
        chosen = json.loads(zero_shot.read_text())["selected"]
    else:
        pytest.skip("no Oxford selection yet")
    assert (detect.DETECTOR, detect.THRESHOLD) == (chosen["selected"] if "selected" in chosen else chosen["detector"],
                                                   chosen["threshold"])


def test_petface_gallery_is_ids_and_scores_only():
    def detect_faces(images):
        return [((8.0, 4.0, 40.0, 36.0), 0.9) for _ in images]

    matcher = serve.Matcher(detect_faces, lambda face: np.eye(6)[0], GALLERY, DOGS, np.zeros(10), 0.1)
    client = TestClient(serve.create_app(matcher, "petface"))
    body = client.post("/match", files={"photo": ("dog.png", _png("red"))}).json()
    assert all(set(m) == {"rank", "dog_id", "score"} for m in body["matches"]) and body["gallery"] == "petface"
    assert "never returned" in body["attribution"] and client.get("/about").json()["returns_gallery_images"] is False


def test_petface_gallery_photos_are_never_served_even_if_paths_are_given(tmp_path):
    photo = tmp_path / "dog0.jpg"
    Image.new("RGB", (8, 8), "red").save(photo)
    matcher = serve.Matcher(lambda images: [None for _ in images], lambda face: np.eye(6)[0], GALLERY, DOGS, np.zeros(10), 0.1)
    client = TestClient(serve.create_app(matcher, "petface", [str(photo)] * 6))
    assert client.get("/gallery/dog0.jpg").status_code == 404
    assert all("image_url" not in m for m in client.post("/match", files={"photo": ("d.png", _png("red"))}).json()["matches"])


def test_dogfacenet_gallery_photos_are_served_with_attribution(tmp_path):
    photos = []
    for i in range(6):
        photos.append(str(tmp_path / f"dog{i}.jpg"))
        Image.new("RGB", (8, 8), "red").save(photos[-1])
    matcher = serve.Matcher(lambda images: [None for _ in images], lambda face: np.eye(6)[0], GALLERY, DOGS, np.zeros(10), 0.1)
    client = TestClient(serve.create_app(matcher, "dogfacenet", photos, ["http://localhost:5173"]))
    body = client.post("/match", files={"photo": ("d.png", _png("red"))}, headers={"Origin": "http://localhost:5173"})
    assert body.headers["access-control-allow-origin"] == "http://localhost:5173"
    first = body.json()["matches"][0]
    assert first["image_url"] == "/gallery/dog0.jpg" and "CC BY 4.0" in body.json()["attribution"]
    assert client.get(first["image_url"]).content[:3] == b"\xff\xd8\xff"  # a JPEG
