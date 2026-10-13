"""Dog face detection and alignment before embedding, for the /match service.
Models load lazily, so the service tests run without torch.

The detector and its score threshold are chosen on Oxford-IIIT Pet half A by recall at
IoU 0.5 against dog head boxes; half B is evaluated once per choice. The first choice
was zero-shot OWL-ViT (results/detection_oxford.json); the current one is a Faster
R-CNN head detector trained on half A (pawmatch.headdet), chosen on half A's internal
validation (results/detection_oxford_trained.json). tests/test_serve.py checks that
the constants match the latest choice. OWL-ViT and OWLv2 (Apache 2.0) are queried
zero-shot with one text, QUERY.
Fixed in advance: the crop, a square around the best box with side CROP_SCALE times
its longer edge, kept inside the photo, at 224 px, and the pre-cropped rule (COVER).
Alignment is position and scale only: the detector gives no landmarks, so faces are
not rotated upright.

Usage: python -m pawmatch.detect CONFIG [speed|oxford]
Writes <results_dir>/serve_detection.json; detector_speed.json with "speed";
results/detection_oxford.json with "oxford" (a held-out evaluation, not DogFaceNet).
"""
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from pawmatch import corrupt, data, evaluate, metrics

DETECTORS = {"owlvit": "google/owlvit-base-patch32", "owlv2": "google/owlv2-base-patch16-ensemble",
             "frcnn_head": "Faster R-CNN MobileNetV3 dog-head detector trained on Oxford half A (pawmatch.headdet)"}
DETECTOR = "frcnn_head"  # chosen on Oxford half A internal validation (results/detection_oxford_trained.json)
QUERY = "a photo of a dog's face"
THRESHOLD = 0.2  # chosen on Oxford half A internal validation
CROP_SCALE = 1.2
ROUTES = ("face_box", "whole_photo_no_box", "whole_photo_box_covers_most")
COVER = 0.5  # a best box over half the photo means the photo is already a face crop
MODEL = "dinov2_vits14"
BATCH = 16
OXFORD = Path("data/oxford_pets")
GRID = (0.01, 0.05, 0.1, 0.2, 0.3)  # thresholds compared on Oxford half A, fixed before the run


class FaceDetector:
    def __init__(self, name=DETECTOR, threshold=THRESHOLD):
        import transformers

        from pawmatch import embed

        prefix = "Owlv2" if name == "owlv2" else "OwlViT"
        self.processor = getattr(transformers, f"{prefix}Processor").from_pretrained(DETECTORS[name])
        self.model = getattr(transformers, f"{prefix}ForObjectDetection").from_pretrained(DETECTORS[name])
        self.model.to(embed.device()).eval()
        self.pads_to_square = name == "owlv2"  # OWLv2 boxes are relative to the padded square image
        self.threshold = threshold

    def __call__(self, images):
        """Best face box (x0, y0, x1, y1) and score per PIL image, or None below THRESHOLD."""
        import torch

        from pawmatch import embed

        texts = [[QUERY]] * len(images)
        sizes = [(max(im.size),) * 2 if self.pads_to_square else im.size[::-1] for im in images]
        with torch.no_grad():
            inputs = self.processor(text=texts, images=images, return_tensors="pt").to(embed.device())
            found = self.processor.post_process_grounded_object_detection(
                self.model(**inputs), threshold=self.threshold, target_sizes=sizes, text_labels=texts)
        out = []
        for r in found:
            if len(r["scores"]) == 0:
                out.append(None)
                continue
            best = int(r["scores"].argmax())
            out.append((tuple(float(v) for v in r["boxes"][best]), float(r["scores"][best])))
        return out


def align(img, box):
    """Square crop around the face box at 224 px: side CROP_SCALE times the box's longer
    edge but no larger than the photo's shorter side, shifted to lie inside the photo,
    so nothing is padded."""
    x0, y0, x1, y1 = box
    w, h = img.size
    side = min(CROP_SCALE * max(x1 - x0, y1 - y0), w, h)
    left = min(max((x0 + x1 - side) / 2, 0), w - side)
    top = min(max((y0 + y1 - side) / 2, 0), h - side)
    return img.crop((round(left), round(top), round(left + side), round(top + side))).resize((224, 224), Image.BICUBIC)


def crop_route(found, size):
    """Design rule for pre-cropped inputs: when no box passed the threshold, or the best
    box covers more than COVER of the photo, the whole photo is the face crop."""
    if found is None:
        return "whole_photo_no_box"
    x0, y0, x1, y1 = found[0]
    w, h = size
    covered = max(0.0, min(x1, w) - max(x0, 0)) * max(0.0, min(y1, h) - max(y0, 0)) / (w * h)
    return "whole_photo_box_covers_most" if covered > COVER else "face_box"


def face_crop(img, found):
    """The image the service embeds, and the route that chose it."""
    route = crop_route(found, img.size)
    return (align(img, found[0]) if route == "face_box" else img), route


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
    commit = evaluate.commit()
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
    return {"hardware": evaluate.hardware(), "code_commit": commit, "batch": BATCH, "images": len(images), "precision": "fp32",
            "rule": "detector chosen on speed alone, a rule set before any detection failure rate was measured (decisions.md notes one earlier single-image spot check)", "detectors": out}


def shares(routes):
    return {r: routes.count(r) / len(routes) for r in ROUTES}


def make_detector(name=DETECTOR, threshold=THRESHOLD):
    """The service detector by name: zero-shot OWL models, or the trained head detector."""
    if name == "frcnn_head":
        from pawmatch import headdet

        return headdet.HeadDetector(threshold)
    return FaceDetector(name, threshold)


class ServicePath:
    """Detect, crop, and embed: the one path for service queries, the service gallery,
    and the val_a photos that calibrate the no-match threshold."""

    def __init__(self, model_name=MODEL):
        from pawmatch import embed

        self.detector = make_detector()
        self.model, self.tf = embed.build(model_name)

    def __call__(self, images):
        """Embeddings, crop routes, and whether a box passed the threshold."""
        found = [f for i in range(0, len(images), BATCH) for f in self.detector(images[i:i + BATCH])]
        crops = [face_crop(im, f) for im, f in zip(images, found)]
        return (embed_faces(self.model, self.tf, [c for c, _ in crops]), [r for _, r in crops],
                np.array([f is not None for f in found]))

    def items(self, items, indices, corruption=None, chunk=256):
        """The same for photos of a split, by index, in chunks."""
        indices = np.asarray(indices)
        parts = [self(_images(items, indices[i:i + chunk], corruption)) for i in range(0, len(indices), chunk)]
        return (np.concatenate([p[0] for p in parts]), [r for p in parts for r in p[1]],
                np.concatenate([p[2] for p in parts]))


def service_index(cfg, path=None):
    """The service gallery (one test photo per dog) and the val_a impostor scores that
    calibrate the confidence label, both through the service path, cached per split,
    detector, and settings. The threshold itself is computed at serving time from the
    actual gallery size (confidence_threshold)."""
    splits, sha = data.load_splits(cfg)
    s = cfg["serve"]
    key = f"{sha}|{DETECTORS[DETECTOR]}|{THRESHOLD}|{CROP_SCALE}|{COVER}|{s['model']}"
    cache = cfg["cache_dir"] / "service_index.npz"
    if cache.exists() and str(np.load(cache)["key"]) == key:
        z = np.load(cache, allow_pickle=False)
        return z["gallery"], z["dogs"], z["impostors"], json.loads(str(z["calibration"]))
    path = path or ServicePath(s["model"])
    items = splits["test"]
    g, _, _ = data.gallery_query(items, cfg["protocol"]["seed"])
    gallery, g_routes, g_found = path.items(items, g)
    va = splits["val_a"]
    va_emb, va_routes, _ = path.items(va, range(len(va)))
    impostors = np.sort(evaluate.pair_scores(va_emb, np.array([d for _, d in va]))[2])
    calibration = {"calibrated_on": "benchmark photos only: val_a photos through the service path",
                   "n_val_a_impostor_pairs": len(impostors), "val_a_routes": shares(va_routes),
                   "gallery_routes": shares(g_routes), "gallery_detection_failure_rate": float(1 - g_found.mean())}
    dogs = np.array([d for _, d in items])[g]
    np.savez(cache, key=key, gallery=gallery, dogs=dogs, impostors=impostors, calibration=json.dumps(calibration))
    return gallery, dogs, impostors, calibration


def confidence_threshold(impostors, gallery_far, gallery_size):
    """Product operating point: the top match is labeled confident when its score is at
    or above t, with 1 - F(t)^N = gallery_far for the actual gallery size N, F being the
    val_a impostor score distribution. Returns (t, per-impostor rate, resolution)."""
    return metrics.threshold_for_gallery(impostors, gallery_far, gallery_size)


def evaluate_detection(cfg):
    """The service path on test photos: detection failure and crop routes for the
    queries, clean and under each SYNTHETIC corruption, rank-1 against the gallery
    (which goes through the same path), and the share of queries that are both
    correct and above the no-match threshold. The threshold's achieved FAR is measured
    on val_b photos, whose dogs are not in the gallery, next to its target."""
    commit = evaluate.commit()  # before the run: later edits must not change the stamp
    splits, sha = data.load_splits(cfg)
    items, seed, n_boot = splits["test"], cfg["protocol"]["seed"], cfg["protocol"]["bootstrap"]
    dogs = np.array([d for _, d in items])
    g, q, _ = data.gallery_query(items, seed)
    path = ServicePath(cfg["serve"]["model"])
    gallery, g_dogs, impostors, calibration = service_index(cfg, path)
    far = cfg["serve"]["confident_gallery_far"]
    threshold, per, res = confidence_threshold(impostors, far, len(gallery))
    vb = splits["val_b"]
    best_b = (path.items(vb, range(len(vb)))[0] @ gallery.T).max(1)
    calibration = {**calibration, "operating_point": "product choice, calibrated on val_a",
                   "target_gallery_far": far, "gallery_size": len(gallery), "threshold": threshold,
                   "per_impostor_far": per, "val_a_resolution": res, "below_resolution": per < res,
                   "achieved_gallery_far_val_b": float((best_b >= threshold).mean()), "n_val_b_photos": len(vb),
                   "oxford_cats": oxford_cats_confident(path, gallery, threshold)}
    qd = dogs[q]
    out = {}
    for condition in (None, *corrupt.CORRUPTIONS):
        emb, routes, found = path.items(items, q, condition)
        hit = evaluate._ranks(emb, qd, gallery, g_dogs) == 1
        best = np.concatenate([(emb[i:i + 1024] @ gallery.T).max(1) for i in range(0, len(emb), 1024)])
        boxed = np.array([r == "face_box" for r in routes])
        out[condition or "clean"] = {
            "n": len(q), "routes": shares(routes),
            "detection_failure_rate": evaluate._score(~found, qd, n_boot, seed),
            "rank1": evaluate._score(hit, qd, n_boot, seed),
            "rank1_face_box_route": evaluate._score(hit[boxed], qd[boxed], n_boot, seed) if boxed.any() else None,
            "correct_and_confident": evaluate._score(hit & (best >= threshold), qd, n_boot, seed),
        }
        print(condition or "clean", f"no box {1 - found.mean():.4f}", f"rank-1 {hit.mean():.4f}")
    return {"dataset": cfg["dataset"]["name"], "split": "test", "split_sha256": sha, "code_commit": commit,
            "detector": DETECTORS[DETECTOR], "query": QUERY, "threshold": THRESHOLD, "crop_scale": CROP_SCALE,
            "cover": COVER, "embedding_model": cfg["serve"]["model"],
            "gallery": "one photo per test dog, through the same detect-and-crop path as queries",
            "corruptions": "SYNTHETIC, queries only, same images as results m2_corruption",
            "confidence_calibration": calibration, "queries": out}


def oxford_species():
    """{photo name: "1" for cat, "2" for dog} from the Oxford-IIIT Pet list."""
    return dict(line.split()[::2][:2] for line in (OXFORD / "annotations" / "list.txt").read_text().splitlines()
                if not line.startswith("#"))


def readable(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def oxford_dogs():
    """Oxford-IIIT Pet dog photos (species 2) with one head box, readable by PIL:
    [(path, (x0, y0, x1, y1))], and the number skipped."""
    species = oxford_species()
    out, skipped = [], 0
    for xml in sorted((OXFORD / "annotations" / "xmls").glob("*.xml")):
        if species.get(xml.stem) != "2":
            continue
        objects = ET.parse(xml).getroot().findall("object")
        path = OXFORD / "images" / f"{xml.stem}.jpg"
        if not readable(path) or len(objects) != 1:
            skipped += 1
            continue
        b = objects[0].find("bndbox")
        out.append((str(path), tuple(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return out, skipped


def oxford_cats_confident(path, gallery, threshold, chunk=256):
    """Non-dog negative set: every readable Oxford-IIIT Pet cat photo goes through the
    service path against the dog gallery. A cat should never be a confident match, so
    the share labeled confident_match is a gallery-level false-accept rate."""
    names = sorted(n for n, sp in oxford_species().items() if sp == "1")
    paths = [p for p in (OXFORD / "images" / f"{n}.jpg" for n in names) if readable(p)]
    best, routes = [], []
    for i in range(0, len(paths), chunk):
        images = []
        for p in paths[i:i + chunk]:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        emb, r, _ = path(images)
        best.append((emb @ gallery.T).max(1))
        routes += r
    best = np.concatenate(best)
    return {"n_photos": len(paths), "n_unreadable": len(names) - len(paths),
            "confident_match_rate": float((best >= threshold).mean()), "routes": shares(routes)}


def oxford_halves():
    """The fixed split of the Oxford dog photos: half A for selection, half B one-shot."""
    dogs, skipped = oxford_dogs()
    order = list(range(len(dogs)))
    random.Random(0).shuffle(order)
    return dogs, skipped, {"A": order[:len(order) // 2], "B": order[len(order) // 2:]}


def best_boxes(detector, paths, batch=BATCH):
    """Best box and score per photo, or None, in batches."""
    found = []
    for i in range(0, len(paths), batch):
        images = []
        for p in paths[i:i + batch]:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        found += detector(images)
    return found


def recall_with_rule(found, sizes, truths, t):
    """Per photo: whether the box the service would use (best box above t, or the whole
    photo under the pre-cropped rule) has IoU >= 0.5 with the head box; whether the
    detector's own box does; and the crop route."""
    hits, raw, routes = [], [], []
    for f, size, truth in zip(found, sizes, truths):
        f = f if f and f[1] > t else None
        route = crop_route(f, size)
        used = f[0] if route == "face_box" else (0, 0, *size)
        hits.append(iou(used, truth) >= 0.5)
        raw.append(f is not None and iou(f[0], truth) >= 0.5)
        routes.append(route)
    return np.array(hits), np.array(raw), routes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def oxford():
    """Recall at IoU 0.5 against Oxford-IIIT Pet dog head boxes, for the box the service
    uses: the detector's best box, or the whole photo under the pre-cropped rule. Each
    detector runs once at the lowest grid threshold; a higher threshold keeps the best
    box only if its score is above it. Half A chooses the detector and threshold by
    that recall, ties going to the higher threshold and then the faster detector
    (OWL-ViT); half B reports the choice with a 95% bootstrap interval over images."""
    commit = evaluate.commit()
    dogs, skipped, halves = oxford_halves()
    sizes = []
    for p, _ in dogs:
        with Image.open(p) as im:
            sizes.append(im.size)
    best = {}
    for name in DETECTORS:
        detector = FaceDetector(name, threshold=min(GRID))
        best[name] = []
        for i in range(0, len(dogs), BATCH):
            images = []
            for p, _ in dogs[i:i + BATCH]:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            best[name] += detector(images)

    def outcome(name, t, idx):
        hits, raw, routes = [], [], []
        for i in idx:
            found = best[name][i] if best[name][i] and best[name][i][1] > t else None
            route = crop_route(found, sizes[i])
            used = found[0] if route == "face_box" else (0, 0, *sizes[i])
            hits.append(iou(used, dogs[i][1]) >= 0.5)
            raw.append(found is not None and iou(found[0], dogs[i][1]) >= 0.5)
            routes.append(route)
        shares = {r: routes.count(r) / len(routes) for r in ("face_box", "whole_photo_no_box", "whole_photo_box_covers_most")}
        return np.array(hits), np.array(raw), shares

    grid = []
    for name in DETECTORS:
        for t in GRID:
            hits, raw, shares = outcome(name, t, halves["A"])
            grid.append({"detector": name, "threshold": t, "recall_at_iou50": float(hits.mean()),
                         "recall_at_iou50_detector_box_only": float(raw.mean()), "routes": shares})
    pick = max(grid, key=lambda r: (r["recall_at_iou50"], r["threshold"], r["detector"] == "owlvit"))
    hits, raw, shares = outcome(pick["detector"], pick["threshold"], halves["B"])
    units = np.arange(len(hits))
    return {
        "dataset": "Oxford-IIIT Pet, dog photos with one head box (CC BY-SA 4.0, Parkhi et al., CVPR 2012)",
        "code_commit": commit, "hardware": evaluate.hardware(), "query": QUERY, "cover": COVER, "iou": 0.5,
        "n_photos": len(dogs), "n_skipped": skipped, "split": "by photo, seed 0", "half_a": len(halves["A"]),
        "half_b": len(halves["B"]), "grid": list(GRID),
        "selection_rule": "highest half-A recall at IoU 0.5 with the pre-cropped rule; ties to the higher threshold, then OWL-ViT",
        "half_a_results": grid, "selected": {"detector": pick["detector"], "model": DETECTORS[pick["detector"]],
                                             "threshold": pick["threshold"]},
        "half_b_result": {
            "recall_at_iou50": {"value": float(hits.mean()),
                                "ci95": metrics.bootstrap_ci(lambda w: np.average(hits, weights=w), units)},
            "recall_at_iou50_detector_box_only": float(raw.mean()), "routes": shares},
    }


def main(config, job="detection"):
    cfg = data.load_config(config)
    if job == "oxford":
        out = Path("results/detection_oxford.json")
        out.write_text(json.dumps(oxford(), indent=1, default=float) + "\n")
        print(out)
        return
    name, result = ("detector_speed", speed(cfg)) if job == "speed" else ("serve_detection", evaluate_detection(cfg))
    out = cfg["results_dir"] / f"{name}.json"
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
