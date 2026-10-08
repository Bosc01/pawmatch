"""Simulated kennel-photo corruptions. SYNTHETIC.

The parameters are fixed in advance at one severity each and are not tuned on any
data. They are applied to query photos only: the found dog photographed at a
kennel. Gallery photos, standing in for owner photos, stay clean.
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage

CORRUPTIONS = ("motion_blur", "low_light", "jpeg", "partial_crop")
BLUR_LENGTH = 15  # pixels, in a random direction
LOW_LIGHT = (0.3, 1.5, 0.03)  # brightness scale, gamma, Gaussian noise std on a 0 to 1 scale
JPEG_QUALITY = 15  # ImageNet-C JPEG severity 3
CROP_AREA = 0.6  # fraction of the image kept, at a random position, resized back
AUGMENT_P = 0.5  # run (c): chance that a training photo gets one random corruption


def corrupt(img, name, rng):
    """Corrupted copy of a PIL RGB image. rng is a numpy Generator."""
    if name == "motion_blur":
        k = np.zeros((BLUR_LENGTH, BLUR_LENGTH))
        k[BLUR_LENGTH // 2] = 1
        k = ndimage.rotate(k, rng.uniform(0, 180), reshape=False, order=1)
        a = np.asarray(img, np.float32)
        a = np.stack([ndimage.convolve(a[..., c], k / k.sum(), mode="reflect") for c in range(3)], -1)
        return Image.fromarray(a.clip(0, 255).round().astype(np.uint8))
    if name == "low_light":
        scale, gamma, noise = LOW_LIGHT
        a = scale * (np.asarray(img, np.float32) / 255) ** gamma
        a = a + rng.normal(0, noise, a.shape)
        return Image.fromarray((a.clip(0, 1) * 255).round().astype(np.uint8))
    if name == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        return Image.open(buf).convert("RGB")
    if name == "partial_crop":
        w, h = img.size
        cw, ch = round(w * CROP_AREA ** 0.5), round(h * CROP_AREA ** 0.5)
        x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        return img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BICUBIC)
    raise ValueError(f"unknown corruption {name}")


class RandomCorruption:
    """Training augmentation for run (c): with probability AUGMENT_P, one corruption
    chosen uniformly."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        if self.rng.random() >= AUGMENT_P:
            return img
        return corrupt(img, CORRUPTIONS[self.rng.integers(len(CORRUPTIONS))], self.rng)
