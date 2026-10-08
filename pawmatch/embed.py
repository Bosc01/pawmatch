"""Backbone embeddings, cached per dataset, model tag, and split.

Usage: python -m pawmatch.embed CONFIG BACKBONE [RUN]
RUN names a fine-tuned checkpoint from pawmatch.train and becomes the cache tag.
"""
import sys

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from pawmatch import data

# CLIP keeps its projection head (the 512-d CLIP image space). The others use pooled
# features: ResNet-50 average pool, DINOv2 CLS token.
BACKBONES = {
    "resnet50_in1k": ("resnet50.tv_in1k", {"num_classes": 0}),
    "clip_vitb16": ("vit_base_patch16_clip_224.openai", {}),
    "dinov2_vits14": ("vit_small_patch14_dinov2.lvd142m", {"num_classes": 0, "img_size": 224}),
}
EMBED_SPLITS = ("val_a", "val_b", "test")


def device():
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
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


@torch.no_grad()
def encode(model, tf, paths, batch_size=64):
    """L2-normalized float32 embeddings, in the order of paths. No loader workers:
    decoding takes 0.6 ms per image against about 14 ms for the forward pass, and
    spawning workers on macOS made a 640-image run four times slower."""
    dev = device()
    model.to(dev).eval()
    loader = DataLoader(Images(paths, tf), batch_size=batch_size)
    return torch.cat([F.normalize(model(x.to(dev)), dim=1).cpu() for x in loader]).numpy()


def main(config, backbone, run=None):
    cfg = data.load_config(config)
    splits, sha = data.load_splits(cfg)
    ckpt = cfg["cache_dir"] / "ckpt" / f"{run}.pt" if run else None
    model, tf = build(backbone, ckpt)
    out_dir = cfg["cache_dir"] / (run or backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in EMBED_SPLITS:
        out = out_dir / f"{s}.npz"
        if not run and out.exists() and str(np.load(out)["sha256"]) == sha:
            print(f"{out} is current, skipping")
            continue
        paths, dogs = zip(*splits[s])
        emb = encode(model, tf, list(paths))
        np.savez(out, emb=emb, paths=np.array(paths), dogs=np.array(dogs), sha256=sha)
        print(f"{out}: {emb.shape}")


if __name__ == "__main__":
    main(*sys.argv[1:])
