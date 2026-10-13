"""Backbone embeddings, cached per dataset, model tag, and split.

Usage: python -m pawmatch.embed CONFIG BACKBONE [--run RUN] [--splits a,b] [--corrupt] [--patches]
--run names a fine-tuned checkpoint from pawmatch.train and becomes the cache tag.
--corrupt also embeds the test split under each SYNTHETIC corruption (test_<name>.npz).
--patches also caches L2-normalized local features of the test split (test_patches.npy).
"""
import argparse

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from pawmatch import corrupt, data

# CLIP keeps its projection head (the 512-d CLIP image space). The others use pooled
# features: ResNet-50 average pool, DINOv2 CLS token.
BACKBONES = {
    "resnet50_in1k": ("resnet50.tv_in1k", {"num_classes": 0}),
    "clip_vitb16": ("vit_base_patch16_clip_224.openai", {}),
    "dinov2_vits14": ("vit_small_patch14_dinov2.lvd142m", {"num_classes": 0, "img_size": 224}),
}
CORRUPTION_SEED = 0


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build(name, checkpoint=None):
    """Model and eval transform. Images are resized to 224x224 with no crop, since
    both datasets are already face crops."""
    arch, kwargs = BACKBONES[name]
    model = timm.create_model(arch, pretrained=checkpoint is None, **kwargs)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["backbone"])
    c = model.pretrained_cfg
    tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode(c["interpolation"])),
                    T.ToTensor(), T.Normalize(c["mean"], c["std"])])
    return model, tf


class Images(Dataset):
    """Images by path. With a corruption name, image i gets that corruption from an rng
    seeded by i, so a corrupted set is the same every time."""

    def __init__(self, paths, tf, corruption=None):
        self.paths, self.tf, self.corruption = paths, tf, corruption

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        if self.corruption:
            im = corrupt.corrupt(im, self.corruption, np.random.default_rng([CORRUPTION_SEED, i]))
        return self.tf(im)


@torch.no_grad()
def encode(model, tf, paths, corruption=None, batch_size=64):
    """L2-normalized float32 embeddings, in the order of paths. No loader workers:
    decoding takes 0.6 ms per image against about 14 ms for the forward pass, and
    spawning workers on macOS made a 640-image run four times slower."""
    dev = device()
    model.to(dev).eval()
    loader = DataLoader(Images(paths, tf, corruption), batch_size=batch_size)
    return torch.cat([F.normalize(model(x.to(dev)), dim=1).cpu() for x in loader]).numpy()


@torch.no_grad()
def encode_patches(model, tf, paths, batch_size=64):
    """L2-normalized local features as float16: DINOv2 patch tokens (256 x 384 at
    224 px) or ResNet-50 final feature-map cells (49 x 2048)."""
    dev = device()
    model.to(dev).eval()
    out = []
    for x in DataLoader(Images(paths, tf), batch_size=batch_size):
        f = model.forward_features(x.to(dev))
        f = f[:, model.num_prefix_tokens:] if f.dim() == 3 else f.flatten(2).transpose(1, 2)
        out.append(F.normalize(f, dim=-1).half().cpu())
    return torch.cat(out).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("backbone", choices=BACKBONES)
    ap.add_argument("--run")
    ap.add_argument("--splits", default="val_a,val_b,test")
    ap.add_argument("--corrupt", action="store_true")
    ap.add_argument("--patches", action="store_true")
    a = ap.parse_args(argv)
    cfg = data.load_config(a.config)
    splits, sha = data.load_splits(cfg)
    model, tf = build(a.backbone, cfg["cache_dir"] / "ckpt" / f"{a.run}.pt" if a.run else None)
    out_dir = cfg["cache_dir"] / (a.run or a.backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, s, None) for s in a.splits.split(",")]
    jobs += [(f"test_{c}", "test", c) for c in corrupt.CORRUPTIONS] if a.corrupt else []
    for name, split, corruption in jobs:
        out = out_dir / f"{name}.npz"
        if not a.run and out.exists() and str(np.load(out)["sha256"]) == sha:
            print(f"{out} is current, skipping")
            continue
        paths, dogs = zip(*splits[split])
        emb = encode(model, tf, list(paths), corruption)
        np.savez(out, emb=emb, paths=np.array(paths), dogs=np.array(dogs), sha256=sha)
        print(f"{out}: {emb.shape}")
    stamp = out_dir / "test_patches.sha256"
    if a.patches and (a.run or not stamp.exists() or stamp.read_text() != sha):
        feats = encode_patches(model, tf, [p for p, _ in splits["test"]])
        np.save(out_dir / "test_patches.npy", feats)
        stamp.write_text(sha)
        print(f"{out_dir / 'test_patches.npy'}: {feats.shape}")


if __name__ == "__main__":
    main()
