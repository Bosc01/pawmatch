"""Cross-attention rerank head over frozen DINOv2 patch tokens (Milestone 2, stage 2).

The head scores a (query, candidate) photo pair from their patch tokens. Both are
projected to D dimensions, the query tokens attend to the candidate tokens in one
cross-attention layer, and the mean of the attended tokens goes through a small MLP to
one logit. It trains on the train split only and listwise: each anchor photo is scored
against one positive (another photo of its dog) and NEG hard negatives (its nearest
photos of other dogs by stage-1 embedding), with softmax cross-entropy over the
1 + NEG candidates. The backbone stays frozen. Outside PetFace at most
train.SMOKE_STEPS steps are allowed. Every value below was fixed before any run.

Usage: python -m pawmatch.xattn CONFIG train   saves data/cache/<ds>/ckpt/xattn.pt
       python -m pawmatch.xattn CONFIG score   scores the stage-1 top 100 for every
                                               test query, for pawmatch.retrieve
"""
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pawmatch import data, embed, evaluate, train

MODEL = "dinov2_vits14"
D, HEADS, NEG, ANCHORS, LR, WEIGHT_DECAY = 128, 4, 7, 16, 1e-4, 0.01


class CrossAttentionHead(nn.Module):
    def __init__(self, dim=384, d=D, heads=HEADS):
        super().__init__()
        self.proj = nn.Linear(dim, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, q, c):
        """q, c: (B, P, dim) patch tokens of B query and B candidate photos -> (B,) logits."""
        q, c = self.proj(q), self.proj(c)
        x = self.norm(q + self.attn(q, c, c, need_weights=False)[0])
        return self.mlp(x.mean(1)).squeeze(1)


def hard_negatives(emb, dogs, k, chunk=1024):
    """For each photo, its k most similar photos of other dogs by stage-1 embedding."""
    out = []
    for i in range(0, len(emb), chunk):
        S = emb[i:i + chunk] @ emb.T
        S[dogs[i:i + chunk, None] == dogs[None, :]] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        out.append(np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1))
    return np.concatenate(out)


def patches(backbone, images, idx):
    """Frozen patch tokens, rounded through float16 like the cached test patches."""
    dev = embed.device()
    with torch.no_grad():
        f = backbone.forward_features(torch.stack([images[i] for i in idx]).to(dev))
        return F.normalize(f[:, backbone.num_prefix_tokens:], dim=-1).half().float()


def fit(cfg, steps):
    splits, _ = data.load_splits(cfg)
    items = splits["train"]
    z = np.load(cfg["cache_dir"] / MODEL / "train.npz")
    if z["paths"].tolist() != [p for p, _ in items]:
        sys.exit("train embeddings do not match the split; run make embed")
    dogs = z["dogs"]
    neg = hard_negatives(z["emb"], dogs, NEG)
    groups = {d: ix for d, ix in data.by_dog(items).items() if len(ix) >= 2}
    anchors_pool = [i for ix in groups.values() for i in ix]
    seed = cfg["dataset"]["seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    backbone, tf = embed.build(MODEL)
    backbone.to(embed.device()).eval().requires_grad_(False)
    images = embed.Images([p for p, _ in items], tf)
    head = CrossAttentionHead().to(embed.device())
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    log = []
    for step in range(steps):
        anchors = rng.sample(anchors_pool, ANCHORS)
        cands = [[rng.choice([j for j in groups[dogs[a]] if j != a]), *neg[a]] for a in anchors]
        q = patches(backbone, images, anchors).repeat_interleave(1 + NEG, 0)
        c = patches(backbone, images, [j for row in cands for j in row])
        logits = head(q, c).view(ANCHORS, 1 + NEG)
        loss = F.cross_entropy(logits, torch.zeros(ANCHORS, dtype=torch.long, device=logits.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append({"loss": loss.item(), "positive_ranked_first": float((logits.argmax(1) == 0).float().mean())})
        print(f"step {step + 1}/{steps} loss {loss.item():.4f}")
    return head, log


@torch.no_grad()
def score(cfg, head):
    """Head logits for each test query against its stage-1 top 100, in stage-1 order."""
    splits, sha = data.load_splits(cfg)
    items = splits["test"]
    g, q, _ = data.gallery_query(items, cfg["protocol"]["seed"])
    g = np.asarray(g)
    emb = np.load(cfg["cache_dir"] / MODEL / "test.npz")["emb"]
    top = evaluate.stage1_top(emb[q], emb[g])
    P = np.load(cfg["cache_dir"] / MODEL / "test_patches.npy", mmap_mode="r")
    dev = embed.device()
    head.to(dev).eval()
    out = np.empty(top.shape, np.float32)
    for i in range(len(q)):
        c = torch.from_numpy(np.asarray(P[g[top[i]]], np.float32)).to(dev)
        qq = torch.from_numpy(np.asarray(P[q[i]], np.float32)).to(dev).expand(len(c), -1, -1)
        out[i] = head(qq, c).cpu().numpy()
    return {"scores": out, "top": top, "sha256": sha}


def main(config, job):
    cfg = data.load_config(config)
    t = cfg["rerank_head"]
    ckpt = cfg["cache_dir"] / "ckpt" / "xattn.pt"
    if job == "train":
        if cfg["dataset"]["name"] != "petface" and not 0 < t["steps"] <= train.SMOKE_STEPS:
            sys.exit(f"full training waits for PetFace; set rerank_head.steps to at most {train.SMOKE_STEPS}")
        if t["steps"] <= 0:
            sys.exit("set rerank_head.steps before the first PetFace run")
        head, log = fit(cfg, t["steps"])
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head": head.cpu().state_dict(), "log": log, "terms": data.model_terms(cfg)}, ckpt)
        print(ckpt)
    else:
        head = CrossAttentionHead()
        head.load_state_dict(torch.load(ckpt, map_location="cpu")["head"])
        out = cfg["cache_dir"] / "xattn_scores.npz"
        np.savez(out, **score(cfg, head))
        print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
