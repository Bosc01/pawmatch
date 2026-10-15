"""PetFace terms (DATA.md): no redistribution. Git must never track PetFace images,
embedding caches, patch features, or checkpoints, nor the private correspondence in
docs/private/, and README and docs show no images except DogFaceNet demo images in
docs/images/dogfacenet/."""
import re
import subprocess
from pathlib import Path

import pytest

DEMO_IMAGES = "docs/images/dogfacenet/"
IMAGE = re.compile(r"\.(png|jpe?g|gif|webp|bmp|tiff?)$", re.I)
DERIVED = re.compile(r"\.(npz|npy|pt|pth|ckpt|safetensors|bin|pkl|h5|onnx)$", re.I)


def tracked():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return out.stdout.splitlines()


def test_no_data_images_caches_or_checkpoints_are_tracked():
    files = tracked()
    assert not [f for f in files if f.startswith("data/")]
    assert not [f for f in files if DERIVED.search(f)]
    code_or_docs = (".py", ".sh", ".toml", ".json", ".md")  # e.g. configs/petface.toml, this test itself
    assert not [f for f in files if "petface" in f.lower() and not f.endswith(code_or_docs)]
    assert not [f for f in files if IMAGE.search(f) and not f.startswith(DEMO_IMAGES)]


def test_private_docs_are_never_tracked():
    assert not [f for f in tracked() if f.startswith("docs/private/")]


def test_readme_and_docs_show_only_dogfacenet_demo_images():
    docs = [f for f in tracked() if f.endswith(".md")]
    for doc in docs:
        text = Path(doc).read_text()
        targets = re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text) + re.findall(r"<img[^>]+src=[\"']([^\"']+)", text)
        assert all(t.lstrip("./").startswith(DEMO_IMAGES) for t in targets), (doc, targets)
