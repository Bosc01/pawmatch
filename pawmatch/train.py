"""Fine-tune DINOv2 ViT-S/14 with ArcFace, with or without batch-hard triplet loss.

Usage: python -m pawmatch.train CONFIG RUN
       python -m pawmatch.train CONFIG precision_test   (DogFaceNet only)

PetFace budget (decisions.md): three runs. Arm (a) "arcface" has no hard-negative
mining and arm (b) "arcface_bh" adds the batch-hard term; together they are the
hard-negative ablation, with the same sampler and schedule. Run (c) "augmented"
repeats the arm with higher val_a rank-1 and adds the Milestone 2 corruption
augmentation. Each run includes its head warm-up. Outside PetFace only smoke-test
steps are allowed.
"""
import json
import math
import random
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from pawmatch import corrupt, data, embed, evaluate

RUNS = {"arcface": 0.0, "arcface_bh": 1.0}  # weight of the batch-hard term
RUN_C = "augmented"
SMOKE_STEPS = 20
DOGS_PER_BATCH, PHOTOS_PER_DOG = 32, 2
# Fixed from standard practice before any PetFace run, not from DogFaceNet numbers
# (decisions.md): head-only warm-up for the first 10% of steps, then the backbone
# unfreezes at a learning rate 100 times below the head.
WARMUP_FRAC = 0.1
LR_BACKBONE, LR_HEAD, WEIGHT_DECAY = 1e-5, 1e-3, 0.05
ARC_MARGIN, TRIPLET_MARGIN = 0.5, 0.3


class ArcFace(nn.Module):
    """Additive angular margin loss (Deng et al., CVPR 2019). The scale follows PetFace
    Appendix B: sqrt(2) * log(C - 1) for C training dogs."""

    def __init__(self, dim, n_classes, margin=ARC_MARGIN):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = math.sqrt(2) * math.log(n_classes - 1), margin

    def forward(self, x, y):
        cos = F.linear(F.normalize(x), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        c = cos.gather(1, y[:, None])
        # Past theta = pi - m, cos(theta + m) stops decreasing, so fall back to a linear penalty.
        target = torch.where(c > math.cos(math.pi - self.m), torch.cos(torch.acos(c) + self.m),
                             c - math.sin(math.pi - self.m) * self.m)
        return F.cross_entropy(self.s * cos.scatter(1, y[:, None], target), y)


def batch_hard(emb, y, margin=TRIPLET_MARGIN):
    """Batch-hard triplet loss (Hermans et al., 2017) on cosine distance."""
    e = F.normalize(emb)
    d = 1 - e @ e.T
    same = y[:, None] == y[None, :]
    hardest_pos = d.masked_fill(~same, -1).max(1).values
    hardest_neg = d.masked_fill(same, 3).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def pk_batches(items, rng):
    """Batches of DOGS_PER_BATCH dogs x PHOTOS_PER_DOG photos, so every photo has a positive."""
    groups = [ix for ix in data.by_dog(items).values() if len(ix) >= PHOTOS_PER_DOG]
    rng.shuffle(groups)
    return [[i for ix in groups[k:k + DOGS_PER_BATCH] for i in rng.sample(ix, PHOTOS_PER_DOG)]
            for k in range(0, len(groups) - DOGS_PER_BATCH + 1, DOGS_PER_BATCH)]


def lr_scales(step, steps):
    """(head, backbone) learning-rate multipliers at a step. The first WARMUP_FRAC of
    steps train only the head; then both follow one cosine decay to zero."""
    warmup = math.ceil(WARMUP_FRAC * steps)
    if step < warmup:
        return 1.0, 0.0
    c = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
    return c, c


def fit(model, tf, items, hard_negative_weight, steps, seed=0, precision="fp32", augment=False, log_every=50):
    """Train model in place for a fixed number of steps. precision is "fp32" or "bf16"
    (autocast for the backbone; losses stay in fp32). augment adds the SYNTHETIC
    corruption augmentation of run (c). Returns one record per step."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    classes = {d: k for k, d in enumerate(sorted({d for _, d in items}))}
    labels = torch.tensor([classes[d] for _, d in items])
    pre = [corrupt.RandomCorruption(seed)] if augment else []
    images = embed.Images([p for p, _ in items], T.Compose([*pre, T.RandomHorizontalFlip(), tf]))
    dev = embed.device()
    head = ArcFace(model.num_features, len(classes)).to(dev)
    model.to(dev).train()
    opt = torch.optim.AdamW([{"params": head.parameters(), "lr": LR_HEAD},
                             {"params": model.parameters(), "lr": LR_BACKBONE}], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: lr_scales(s, steps)[0], lambda s: lr_scales(s, steps)[1]])
    batches, log = [], []
    for step in range(steps):
        start = time.perf_counter()
        frozen = lr_scales(step, steps)[1] == 0
        model.requires_grad_(not frozen)
        batches = batches or pk_batches(items, rng)
        ix = batches.pop()
        x, y = torch.stack([images[i] for i in ix]).to(dev), labels[ix].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=precision == "bf16"), torch.set_grad_enabled(not frozen):
            e = model(x).float()
        arc, bh = head(e, y), batch_hard(e, y)
        loss = arc + hard_negative_weight * bh
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss is {loss.item()} at step {step + 1}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        log.append({"phase": "head" if frozen else "full", "arcface": arc.item(), "batch_hard": bh.item(),
                    "seconds": time.perf_counter() - start})
        if (step + 1) % log_every == 0 or step + 1 == steps:
            print(f"step {step + 1}/{steps} {log[-1]['phase']} arcface {arc.item():.4f} batch_hard {bh.item():.4f}")
    return log


def precision_test(cfg):
    """Speed check on DogFaceNet only: SMOKE_STEPS steps of arm (b) in fp32 and in bf16
    mixed precision from the same seed. The first step of each phase is excluded from
    the medians, since it includes one-time setup."""
    if cfg["dataset"]["name"] == "petface":
        sys.exit("the precision test runs on DogFaceNet only")
    splits, _ = data.load_splits(cfg)
    log = {}
    for p in ("fp32", "bf16"):
        model, tf = embed.build("dinov2_vits14")
        log[p] = fit(model, tf, splits["train"], RUNS["arcface_bh"], SMOKE_STEPS, cfg["dataset"]["seed"], p, log_every=1)

    def median(p, phase):
        times = [r["seconds"] for r in log[p] if r["phase"] == phase][1:]
        return statistics.median(times) if times else None

    out = cfg["results_dir"] / "precision_test.json"
    out.write_text(json.dumps({
        "dataset": cfg["dataset"]["name"], "purpose": "speed only", "hardware": evaluate.hardware(),
        "code_commit": evaluate.commit(), "steps": SMOKE_STEPS, "warmup_steps": math.ceil(WARMUP_FRAC * SMOKE_STEPS),
        "batch_size": DOGS_PER_BATCH * PHOTOS_PER_DOG, "seed": cfg["dataset"]["seed"],
        "median_seconds_per_step": {p: {ph: median(p, ph) for ph in ("head", "full")} for p in log},
        "log": log,
    }, indent=1) + "\n")
    print(out)


def main(config, run):
    cfg = data.load_config(config)
    t, name = cfg["train"], cfg["dataset"]["name"]
    if run == "precision_test":
        return precision_test(cfg)
    arm = t.get("run_c_arm") if run == RUN_C else run
    if run == RUN_C and arm not in RUNS:
        sys.exit("set train.run_c_arm to the arm with higher val_a rank-1 before run (c)")
    if arm not in RUNS:
        sys.exit(f"unknown run {run}")
    if name != "petface" and t["steps"] > SMOKE_STEPS:
        sys.exit(f"full fine-tunes are reserved for PetFace; set train.steps to at most {SMOKE_STEPS}")
    out = cfg["cache_dir"] / "ckpt" / f"{run}.pt"
    if name == "petface" and out.exists():
        sys.exit(f"{out} exists; each PetFace run is used once (decisions.md)")
    splits, _ = data.load_splits(cfg)
    model, tf = embed.build("dinov2_vits14")
    log = fit(model, tf, splits["train"], RUNS[arm], t["steps"], cfg["dataset"]["seed"], t["precision"],
              augment=run == RUN_C, log_every=1 if t["steps"] <= SMOKE_STEPS else 50)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.cpu().state_dict(), "run": run, "arm": arm, "log": log}, out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
