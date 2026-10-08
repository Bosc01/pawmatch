"""Fine-tune DINOv2 ViT-S/14 with ArcFace, with or without batch-hard triplet loss.

Usage: python -m pawmatch.train CONFIG RUN
RUN "arcface" has no hard-negative mining; "arcface_bh" adds the batch-hard term.
That pair is the hard-negative ablation. Both use the same P x K sampler, so the
loss is the only difference. Outside PetFace only smoke-test steps are allowed.
"""
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from pawmatch import data, embed

RUNS = {"arcface": 0.0, "arcface_bh": 1.0}  # weight of the batch-hard term
SMOKE_STEPS = 20
DOGS_PER_BATCH, PHOTOS_PER_DOG = 32, 2
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


def fit(model, tf, items, hard_negative_weight, epochs, max_steps=None, seed=0, log_every=50):
    """Train model in place on items. Returns per-step losses."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    classes = {d: k for k, d in enumerate(sorted({d for _, d in items}))}
    labels = torch.tensor([classes[d] for _, d in items])
    dev = embed.device()
    head = ArcFace(model.num_features, len(classes)).to(dev)
    model.to(dev).train()
    total = epochs * len(pk_batches(items, random.Random(seed)))
    total = min(total, max_steps) if max_steps else total
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": LR_BACKBONE},
                             {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    images = embed.Images([p for p, _ in items], T.Compose([T.RandomHorizontalFlip(), tf]))
    losses = []
    for _ in range(epochs):
        batches = pk_batches(items, rng)
        for ix, x in zip(batches, DataLoader(images, batch_sampler=batches)):
            y = labels[ix].to(dev)
            e = model(x.to(dev))
            arc, bh = head(e, y), batch_hard(e, y)
            loss = arc + hard_negative_weight * bh
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss is {loss.item()} at step {len(losses) + 1}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append({"arcface": arc.item(), "batch_hard": bh.item()})
            if len(losses) % log_every == 0 or len(losses) == total:
                print(f"step {len(losses)}/{total} arcface {arc.item():.4f} batch_hard {bh.item():.4f}")
            if len(losses) >= total:
                return losses
    return losses


def main(config, run):
    cfg = data.load_config(config)
    t = cfg["train"]
    if cfg["dataset"]["name"] != "petface" and not 0 < t.get("max_steps", 0) <= SMOKE_STEPS:
        sys.exit(f"full fine-tunes are reserved for PetFace; set train.max_steps to at most {SMOKE_STEPS}")
    splits, _ = data.load_splits(cfg)
    model, tf = embed.build("dinov2_vits14")
    losses = fit(model, tf, splits["train"], RUNS[run], t["epochs"], t.get("max_steps"),
                 cfg["dataset"]["seed"], log_every=1 if t.get("max_steps") else 50)
    out = cfg["cache_dir"] / "ckpt" / f"{run}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.cpu().state_dict(), "run": run, "losses": losses}, out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
