"""Single-class dog-head detector for the /match service, trained on Oxford-IIIT Pet
half A only (CC BY-SA 4.0 data; DATA.md).

torchvision Faster R-CNN with a MobileNetV3-Large FPN backbone (BSD-3 code, COCO
pretrained weights), with its box predictor replaced for one class. The training
settings are those of the torchvision object detection fine-tuning tutorial, fixed
before the run: SGD, learning rate 0.005, momentum 0.9, weight decay 0.0005, the
learning rate divided by 10 every 3 epochs, 10 epochs, batch 2, horizontal flips, and
a linear warm-up over the first epoch. Half A is split 80/20 by photo into train and
internal validation. There is no checkpoint selection: the final weights are used.

Selection, on internal validation only: the score threshold from detect.GRID, and then
the service detector, this one or OWL-ViT at its Oxford half-A threshold, both by
recall at IoU 0.5 under the pre-cropped rule, ties going to the higher threshold and
then to OWL-ViT. Half B is evaluated once, for this detector, as a second look at half B.

Usage: python -m pawmatch.headdet train | evaluate
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import hflip, to_tensor

from pawmatch import detect, embed, evaluate, metrics

EPOCHS, BATCH, LR, MOMENTUM, WEIGHT_DECAY, LR_STEP, LR_GAMMA = 10, 2, 0.005, 0.9, 0.0005, 3, 0.1
VAL_FRAC, SEED = 0.2, 1
EVAL_BATCH = 4  # inference batch; batches of 16 photos of mixed sizes grew MPS memory past RAM
CKPT = Path("data/cache/headdet/frcnn_mobilenet_v3_head.pt")


def build():
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    return model


def half_a_split():
    """Half A of the Oxford dog photos, split 80/20 by photo: (dogs, train, internal val)."""
    dogs, _, halves = detect.oxford_halves()
    a = list(halves["A"])
    random.Random(SEED).shuffle(a)
    n_val = round(VAL_FRAC * len(a))
    return dogs, a[n_val:], a[:n_val]


class HeadDetector:
    """Same interface as detect.FaceDetector: best (box, score) per photo, or None."""

    def __init__(self, threshold, checkpoint=CKPT, device=None):
        self.device = device or embed.device()
        self.model = build()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model.to(self.device).eval()
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, images):
        out = self.model([to_tensor(im).to(self.device) for im in images])
        out = [{k: v.cpu() for k, v in o.items()} for o in out]
        if self.device.type == "mps":
            torch.mps.empty_cache()  # variable photo sizes otherwise grow the MPS cache past RAM
        found = []
        for o in out:
            keep = o["scores"] > self.threshold
            if not keep.any():
                found.append(None)
                continue
            best = int(o["scores"][keep].argmax())
            found.append((tuple(float(v) for v in o["boxes"][keep][best]), float(o["scores"][keep][best])))
        return found


def train():
    dogs, train_idx, _ = half_a_split()
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    dev = embed.device()
    model = build().to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_idx) // BATCH
    warmup_iters = min(1000, steps_per_epoch - 1)
    log = []
    for epoch in range(EPOCHS):
        order = list(train_idx)
        rng.shuffle(order)
        losses = []
        for k in range(steps_per_epoch):
            lr = LR * LR_GAMMA ** (epoch // LR_STEP)
            if epoch == 0:  # linear warm-up from 0.001 x, as in the tutorial
                lr *= 0.001 + (1 - 0.001) * min(k, warmup_iters) / warmup_iters
            for group in opt.param_groups:
                group["lr"] = lr
            images, targets = [], []
            for i in order[k * BATCH:(k + 1) * BATCH]:
                path, (x0, y0, x1, y1) = dogs[i]
                with Image.open(path) as im:
                    img = to_tensor(im.convert("RGB"))
                if rng.random() < 0.5:
                    w = img.shape[-1]
                    img, (x0, x1) = hflip(img), (w - x1, w - x0)
                images.append(img.to(dev))
                targets.append({"boxes": torch.tensor([[x0, y0, x1, y1]], device=dev),
                                "labels": torch.ones(1, dtype=torch.int64, device=dev)})
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss {loss.item()} in epoch {epoch + 1}, step {k + 1}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "lr_end": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch + 1}/{EPOCHS} mean loss {np.mean(losses):.4f}")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), CKPT)
    CKPT.with_suffix(".json").write_text(json.dumps({"log": log, "train_photos": len(train_idx)}, indent=1) + "\n")
    print(CKPT)


def evaluate_head():
    commit = evaluate.commit()
    dogs, train_idx, val_idx = half_a_split()
    _, _, halves = detect.oxford_halves()
    sizes = {}
    for i in range(len(dogs)):
        with Image.open(dogs[i][0]) as im:
            sizes[i] = im.size

    def found_on(detector, idx):
        return detect.best_boxes(detector, [dogs[i][0] for i in idx], batch=EVAL_BATCH)

    def score(found, idx, t):
        return detect.recall_with_rule(found, [sizes[i] for i in idx], [dogs[i][1] for i in idx], t)

    head_val = found_on(HeadDetector(min(detect.GRID)), val_idx)
    grid = [{"threshold": t, "recall_at_iou50": float(score(head_val, val_idx, t)[0].mean())} for t in detect.GRID]
    t_head = max(grid, key=lambda r: (r["recall_at_iou50"], r["threshold"]))["threshold"]
    oxford = json.loads(Path("results/detection_oxford.json").read_text())
    t_owl = oxford["selected"]["threshold"]
    owl_val = found_on(detect.FaceDetector("owlvit", threshold=t_owl), val_idx)
    candidates = {"frcnn_head": float(score(head_val, val_idx, t_head)[0].mean()),
                  "owlvit": float(score(owl_val, val_idx, t_owl)[0].mean())}
    service = max(candidates, key=lambda k: (candidates[k], k == "owlvit"))

    b = halves["B"]
    hits, raw, routes = score(found_on(HeadDetector(t_head), b), b, t_head)
    result = {
        "dataset": oxford["dataset"], "code_commit": commit, "hardware": evaluate.hardware(),
        "model": "torchvision fasterrcnn_mobilenet_v3_large_fpn, COCO weights, one-class head",
        "training": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
                     "lr_step_epochs": LR_STEP, "lr_gamma": LR_GAMMA, "source": "torchvision detection fine-tuning tutorial",
                     "log": json.loads(CKPT.with_suffix(".json").read_text())["log"]},
        "half_a_split": {"train": len(train_idx), "internal_val": len(val_idx), "seed": SEED},
        "internal_val_threshold_grid": grid, "selected_threshold": t_head,
        "internal_val_service_choice": {"recall_at_iou50": candidates, "selected": service,
                                        "threshold": t_head if service == "frcnn_head" else t_owl},
        "half_b_result": {
            "look": "second look at half B (the first was the OWL-ViT choice in detection_oxford.json)",
            "recall_at_iou50": {"value": float(hits.mean()), "ci95": metrics.bootstrap_ci(
                lambda w: np.average(hits, weights=w), np.arange(len(hits)))},
            "recall_at_iou50_detector_box_only": float(raw.mean()), "routes": detect.shares(routes),
            "baseline_owlvit_half_b": oxford["half_b_result"]["recall_at_iou50"]},
    }
    out = Path("results/detection_oxford_trained.json")
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(out, json.dumps(result["internal_val_service_choice"]), f"half B {hits.mean():.4f}")


if __name__ == "__main__":
    {"train": train, "evaluate": evaluate_head}[sys.argv[1]]()
