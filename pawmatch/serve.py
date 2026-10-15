"""PawMatch demo service: POST /match.

A found-dog photo goes through dog face detection and alignment (pawmatch.detect),
embedding, and exact search over a gallery of one photo per dog from the test split.
The confidence label uses the open-set rule from Milestone 2 at a product operating
point: a threshold from the val_a impostor score distribution at the gallery size. Photos are processed in memory
and never stored, and no personal data is collected. This is a demo: the gallery is a
public research dataset, not real lost-dog reports. The optional description is
accepted but not used until the Milestone 3 text model exists.

Run: make serve CONFIG=configs/dogfacenet.toml
"""
import io
import os

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from pawmatch import detect

MAX_BYTES = 10 * 2**20
WHOLE_PHOTO = {
    "whole_photo_no_box": "No separate dog face was found, so the whole photo was matched.",
    "whole_photo_box_covers_most": "The dog's face fills most of the photo, so the whole photo was matched.",
}
NOT_IMAGE = "The upload is not a readable image."
CONFIDENCE_CALIBRATION = ("The confidence label is a product choice calibrated on benchmark photos only "
                          "(DogFaceNet val_a photos through the same detect-and-crop path), not on real "
                          "found-dog photos. All five candidates are always returned.")
DEMO_NOTE = "Demo only. The gallery is a public research dataset, not real lost-dog reports."
ATTRIBUTION = {
    "dogfacenet": ("Gallery photos: DogFaceNet test split, CC BY 4.0. Guillaume Mougeot, 'The DogFaceNet "
                   "datasets', Zenodo, 2024, doi:10.5281/zenodo.12578449."),
    # PetFace terms (DATA.md): view-only display, so gallery entries are IDs and scores, never images.
    "petface": ("Gallery: PetFace test split (Shinoda and Shiohara, ECCV 2024), non-commercial research use "
                "only. Entries are IDs and scores; PetFace images are never returned."),
}
MATCH_FIELDS = ("rank", "dog_id", "score")
# Only DogFaceNet gallery photos may be shown (CC BY 4.0, attributed). PetFace display
# terms: PetFace entries stay IDs and scores, and no PetFace image is ever served.
IMAGE_GALLERIES = {"dogfacenet"}


class Matcher:
    """detect(images) -> [(box, score) or None]; embed_face(PIL image) -> unit vector.
    The confidence threshold is computed here, at serving time, from the actual gallery
    size and the val_a impostor scores (detect.confidence_threshold)."""

    def __init__(self, detect_faces, embed_face, gallery, gallery_dogs, impostors, gallery_far, top_k=5):
        self.detect_faces, self.embed_face = detect_faces, embed_face
        self.gallery, self.dogs, self.gallery_far, self.top_k = gallery, gallery_dogs, gallery_far, top_k
        self.threshold = detect.confidence_threshold(impostors, gallery_far, len(gallery))[0]

    def __call__(self, img):
        found = self.detect_faces([img])[0]
        crop, route = detect.face_crop(img, found)
        scores = self.gallery @ self.embed_face(crop)
        top = np.argsort(-scores, kind="stable")[:self.top_k]
        return {
            "matches": [dict(zip(MATCH_FIELDS, (r + 1, str(self.dogs[i]), float(scores[i])))) for r, i in enumerate(top)],
            "confidence": "confident_match" if scores[top[0]] >= self.threshold else "uncertain",
            "confidence_threshold": self.threshold,
            "operating_point": {"gallery_far": self.gallery_far, "gallery_size": len(self.gallery)},
            "confidence_calibration": CONFIDENCE_CALIBRATION,
            "face": {"route": route, "note": WHOLE_PHOTO.get(route),
                     "box": None if found is None else [round(v, 1) for v in found[0]],
                     "score": None if found is None else found[1]},
        }


def create_app(matcher, dataset="dogfacenet", gallery_paths=None, allowed_origins=()):
    app = FastAPI(title="PawMatch demo")
    attribution = ATTRIBUTION[dataset]
    images = dict(zip(map(str, matcher.dogs), gallery_paths or [])) if dataset in IMAGE_GALLERIES else {}
    if allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(allowed_origins), allow_methods=["GET", "POST"],
                           allow_headers=["*"])

    @app.get("/gallery/{dog_id}.jpg")
    def gallery_image(dog_id: str):
        if dog_id not in images:
            raise HTTPException(404, "No displayable gallery photo.")
        return FileResponse(images[dog_id], media_type="image/jpeg")

    @app.get("/about")
    def about():
        return {"note": DEMO_NOTE, "attribution": attribution, "gallery": dataset, "stores_photos": False,
                "returns_gallery_images": bool(images)}

    @app.post("/match")
    async def match(photo: UploadFile = File(...), description: str | None = Form(None)):
        data = await photo.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"The photo is larger than {MAX_BYTES // 2**20} MB.")
        try:
            with Image.open(io.BytesIO(data)) as im:
                img = im.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, NOT_IMAGE)
        result = matcher(img)
        for m in result["matches"]:
            if m["dog_id"] in images:
                m["image_url"] = f"/gallery/{m['dog_id']}.jpg"
        return {**result, "gallery": dataset, "description_used": False, "note": DEMO_NOTE, "attribution": attribution}

    return app


def build_app():
    """Factory for uvicorn: reads PAWMATCH_CONFIG, loads the models, and builds or loads
    the gallery and no-match threshold through the service path."""
    from pawmatch import data

    cfg = data.load_config(os.environ.get("PAWMATCH_CONFIG", "configs/dogfacenet.toml"))
    path = detect.service_path(cfg)
    gallery, dogs, impostors, _ = detect.service_index(cfg, path)
    splits, _ = data.load_splits(cfg)
    g, _, _ = data.gallery_query(splits["test"], cfg["protocol"]["seed"])
    assert [splits["test"][i][1] for i in g] == list(dogs), "gallery order changed; rebuild the service index"
    origins = [o for o in os.environ.get("PAWMATCH_ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o]
    return create_app(Matcher(path.detector, lambda face: detect.embed_faces(path.model, path.tf, [face])[0],
                              gallery, dogs, impostors, cfg["serve"]["confident_gallery_far"], cfg["serve"]["top_k"]),
                      cfg["dataset"]["name"], [splits["test"][i][0] for i in g], origins)
