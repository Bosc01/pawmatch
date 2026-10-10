"""PawMatch demo service: POST /match.

A found-dog photo goes through dog face detection and alignment (pawmatch.detect),
embedding, and exact search over a gallery of one photo per dog from the test split.
The no-match flag uses the open-set rule from Milestone 2: a threshold from the
val_a impostor score distribution at the gallery size. Photos are processed in memory
and never stored, and no personal data is collected. This is a demo: the gallery is a
public research dataset, not real lost-dog reports. The optional description is
accepted but not used until the Milestone 3 text model exists.

Run: make serve CONFIG=configs/dogfacenet.toml
"""
import io
import os

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from pawmatch import detect

MAX_BYTES = 10 * 2**20
WHOLE_PHOTO = {
    "whole_photo_no_box": "No separate dog face was found, so the whole photo was matched.",
    "whole_photo_box_covers_most": "The dog's face fills most of the photo, so the whole photo was matched.",
}
NOT_IMAGE = "The upload is not a readable image."
DEMO_NOTE = "Demo only. The gallery is a public research dataset, not real lost-dog reports."
ATTRIBUTION = ("Gallery photos: DogFaceNet test split, CC BY 4.0. Guillaume Mougeot, 'The DogFaceNet "
               "datasets', Zenodo, 2024, doi:10.5281/zenodo.12578449.")


class Matcher:
    """detect(images) -> [(box, score) or None]; embed_face(PIL image) -> unit vector."""

    def __init__(self, detect_faces, embed_face, gallery, gallery_dogs, threshold, top_k=5):
        self.detect_faces, self.embed_face = detect_faces, embed_face
        self.gallery, self.dogs, self.threshold, self.top_k = gallery, gallery_dogs, threshold, top_k

    def __call__(self, img):
        found = self.detect_faces([img])[0]
        crop, route = detect.face_crop(img, found)
        scores = self.gallery @ self.embed_face(crop)
        top = np.argsort(-scores, kind="stable")[:self.top_k]
        return {
            "matches": [{"rank": r + 1, "dog_id": str(self.dogs[i]), "score": float(scores[i])} for r, i in enumerate(top)],
            "no_match": bool(scores[top[0]] < self.threshold),
            "no_match_threshold": self.threshold,
            "face": {"route": route, "note": WHOLE_PHOTO.get(route),
                     "box": None if found is None else [round(v, 1) for v in found[0]],
                     "score": None if found is None else found[1]},
        }


def create_app(matcher):
    app = FastAPI(title="PawMatch demo")

    @app.get("/about")
    def about():
        return {"note": DEMO_NOTE, "attribution": ATTRIBUTION, "stores_photos": False}

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
        return {**matcher(img), "description_used": False, "note": DEMO_NOTE, "attribution": ATTRIBUTION}

    return app


def build_app():
    """Factory for uvicorn: reads PAWMATCH_CONFIG and loads the models and gallery."""
    from pawmatch import data, embed, metrics, retrieve

    cfg = data.load_config(os.environ.get("PAWMATCH_CONFIG", "configs/dogfacenet.toml"))
    s = cfg["serve"]
    splits, _ = data.load_splits(cfg)
    g, _, _ = data.gallery_query(splits["test"], cfg["protocol"]["seed"])
    test = np.load(cfg["cache_dir"] / s["model"] / "test.npz")
    va = np.load(cfg["cache_dir"] / s["model"] / "val_a.npz")
    threshold = metrics.threshold_for_gallery(retrieve._pairs(va["emb"], va["dogs"])[2], s["no_match_far"], len(g))[0]
    model, tf = embed.build(s["model"])
    return create_app(Matcher(detect.FaceDetector(), lambda face: detect.embed_faces(model, tf, [face])[0],
                              test["emb"][g], test["dogs"][g], threshold, s["top_k"]))
