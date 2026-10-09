"""Dog face detection and alignment before embedding, for the /match service.
Models load lazily, so the service tests run without torch.

OWL-ViT (google/owlvit-base-patch32, Apache 2.0) is queried zero-shot with one text,
QUERY. It was chosen over OWLv2 on speed alone, a rule set before any detection
result was measured (results/dev/detector_speed.json). Fixed in advance and not tuned
on any data: the score threshold 0.1, which is the transformers default for OWL
detection, and the crop, a square centred on the best box with side CROP_SCALE times
its longer edge, resized to 224 px. Alignment is position and scale only: the detector
gives no landmarks, so faces are not rotated upright.

Usage: python -m pawmatch.detect CONFIG [speed]
Writes <results_dir>/serve_detection.json, or detector_speed.json with "speed".
"""
import json
import sys
import time

import numpy as np
from PIL import Image

from pawmatch import corrupt, data, evaluate

DETECTORS = {"owlvit": "google/owlvit-base-patch32", "owlv2": "google/owlv2-base-patch16-ensemble"}
DETECTOR = "owlvit"
QUERY = "a photo of a dog's face"
THRESHOLD = 0.1
CROP_SCALE = 1.2
MODEL = "dinov2_vits14"
BATCH = 16


class FaceDetector:
    def __init__(self, name=DETECTOR):
        import transformers

        from pawmatch import embed

        prefix = "Owlv2" if name == "owlv2" else "OwlViT"
        self.processor = getattr(transformers, f"{prefix}Processor").from_pretrained(DETECTORS[name])
        self.model = getattr(transformers, f"{prefix}ForObjectDetection").from_pretrained(DETECTORS[name])
        self.model.to(embed.device()).eval()
        self.pads_to_square = name == "owlv2"  # OWLv2 boxes are relative to the padded square image

    def __call__(self, images):
        """Best face box (x0, y0, x1, y1) and score per PIL image, or None below THRESHOLD."""
        import torch

        from pawmatch import embed

        texts = [[QUERY]] * len(images)
        sizes = [(max(im.size),) * 2 if self.pads_to_square else im.size[::-1] for im in images]
        with torch.no_grad():
            inputs = self.processor(text=texts, images=images, return_tensors="pt").to(embed.device())
            found = self.processor.post_process_grounded_object_detection(
                self.model(**inputs), threshold=THRESHOLD, target_sizes=sizes, text_labels=texts)
        out = []
        for r in found:
            if len(r["scores"]) == 0:
                out.append(None)
                continue
            best = int(r["scores"].argmax())
            out.append((tuple(float(v) for v in r["boxes"][best]), float(r["scores"][best])))
        return out


def align(img, box):
    """Square crop centred on the face box, side CROP_SCALE times its longer edge, at
    224 px. Parts of the square outside the photo are filled with black."""
    x0, y0, x1, y1 = box
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, CROP_SCALE * max(x1 - x0, y1 - y0) / 2
    crop = img.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))
    return crop.resize((224, 224), Image.BICUBIC)


def embed_faces(model, tf, faces):
    """L2-normalized embeddings of aligned face crops."""
    import torch
    import torch.nn.functional as F

    from pawmatch import embed

    dev = embed.device()
    model.to(dev).eval()
    with torch.no_grad():
        return torch.cat([F.normalize(model(torch.stack([tf(f) for f in faces[i:i + 64]]).to(dev)), dim=1).cpu()
                          for i in range(0, len(faces), 64)]).numpy()


def _images(items, indices, corruption):
    """Photos as the embedding cache saw them: corruption i uses the rng seeded by i."""
    from pawmatch import embed

    loader = embed.Images([p for p, _ in items], lambda im: im, corruption)
    return [loader[i] for i in indices]


def speed(cfg):
    """Detector time per image on 32 DogFaceNet photos, batch 16; second of two passes."""
    splits, _ = data.load_splits(cfg)
    images = _images(splits["test"], range(32), None)
    out = {}
    for name in DETECTORS:
        detector = FaceDetector(name)
        for _ in range(2):
            start = time.perf_counter()
            for i in range(0, len(images), BATCH):
                detector(images[i:i + BATCH])
            ms = 1000 * (time.perf_counter() - start) / len(images)
        out[name] = {"model": DETECTORS[name], "ms_per_image": ms}
    return {"hardware": evaluate.hardware(), "batch": BATCH, "images": len(images), "precision": "fp32",
            "rule": "detector chosen on speed alone, before any detection result was measured", "detectors": out}


def evaluate_detection(cfg):
    """Detection failure rate on the test queries, clean and under each SYNTHETIC
    corruption, plus rank-1 through the full service path (detect, align, embed,
    search the one-photo-per-dog gallery). A detection failure counts as a miss."""
    splits, sha = data.load_splits(cfg)
    items, seed, n_boot = splits["test"], cfg["protocol"]["seed"], cfg["protocol"]["bootstrap"]
    dogs = np.array([d for _, d in items])
    g, q, _ = data.gallery_query(items, seed)
    gallery = np.load(cfg["cache_dir"] / MODEL / "test.npz")["emb"][g]
    from pawmatch import embed

    detector, (model, tf) = FaceDetector(), embed.build(MODEL)

    def run(indices, corruption, chunk=256):
        ok, hit = np.zeros(len(indices), bool), np.zeros(len(indices), bool)
        for start in range(0, len(indices), chunk):
            part = np.asarray(indices[start:start + chunk])
            images = _images(items, part, corruption)
            found = [f for i in range(0, len(images), BATCH) for f in detector(images[i:i + BATCH])]
            got = np.array([f is not None for f in found])
            ok[start:start + len(part)] = got
            if got.any():
                faces = [align(im, f[0]) for im, f in zip(images, found) if f is not None]
                ranks = evaluate._ranks(embed_faces(model, tf, faces), dogs[part][got], gallery, dogs[g])
                hit[start:start + len(part)][got] = ranks == 1
        return ok, hit

    qd = dogs[q]
    ok, _ = run(g, None)
    out = {"gallery_photos": {"n": len(g), "detection_failure_rate": float(1 - ok.mean())}, "queries": {}}
    for condition in (None, *corrupt.CORRUPTIONS):
        ok, hit = run(q, condition)
        out["queries"][condition or "clean"] = {
            "n": len(q),
            "detection_failure_rate": evaluate._score(~ok, qd, n_boot, seed),
            "rank1_end_to_end": evaluate._score(hit, qd, n_boot, seed),
            "rank1_among_detected": evaluate._score(hit[ok], qd[ok], n_boot, seed) if ok.any() else None,
        }
        print(condition or "clean", f"failure {1 - ok.mean():.4f}", f"end-to-end rank-1 {hit.mean():.4f}")
    return {"dataset": cfg["dataset"]["name"], "split": "test", "split_sha256": sha, "code_commit": evaluate.commit(),
            "detector": DETECTORS[DETECTOR], "query": QUERY, "threshold": THRESHOLD, "crop_scale": CROP_SCALE,
            "embedding_model": MODEL, "gallery": "one photo per test dog, embedded without detection (already face crops)",
            "corruptions": "SYNTHETIC, queries only, same images as results m2_corruption", **out}


def main(config, job="detection"):
    cfg = data.load_config(config)
    name, result = ("detector_speed", speed(cfg)) if job == "speed" else ("serve_detection", evaluate_detection(cfg))
    out = cfg["results_dir"] / f"{name}.json"
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
