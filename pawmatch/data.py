"""Dataset loaders, frozen identity-disjoint splits, and evaluation protocols.

Every dataset loads to the same shape, {split: [(image_path, dog_id), ...]}, for
the splits train, val_a, val_b, test. val_a calibrates the no-match threshold and
val_b trains the learned ranker. Switching datasets is a config change only.
"""
import csv
import hashlib
import json
import random
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val_a", "val_b", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
FROZEN_DIR = Path("splits")
# PetFace authors' terms, received 2026-10-14 (DATA.md): stamped into every model trained on PetFace.
PETFACE_MODEL_TERMS = "Trained on PetFace: non-commercial use only; do not redistribute (DATA.md, terms of 2026-10-14)."


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    name = cfg["dataset"]["name"]
    # Only PetFace results are reportable. Everything else is development output.
    cfg["results_dir"] = Path("results" if name == "petface" else "results/dev")
    cfg["cache_dir"] = Path("data/cache") / name
    return cfg


def model_terms(cfg):
    """Usage terms for a checkpoint trained under cfg: PetFace-trained models are non-commercial."""
    return PETFACE_MODEL_TERMS if cfg["dataset"]["name"] == "petface" else None


def dog_of(path):
    """Both datasets store one folder per dog: .../<dog_id>/<image>."""
    return Path(path).parent.name


def by_dog(items):
    """{dog_id: [indices into items]}, sorted by dog id."""
    out = defaultdict(list)
    for i, (_, d) in enumerate(items):
        out[d].append(i)
    return dict(sorted(out.items()))


def _half(dogs, seed):
    dogs = sorted(dogs)
    random.Random(seed).shuffle(dogs)
    return set(dogs[: len(dogs) // 2])


def _petface(ds):
    """Official PetFace dog split files. The validation dogs are halved into val_a and val_b."""
    images, split = Path(ds["images"]), Path(ds["split_dir"])
    with open(split / "train.csv", newline="") as f:
        files = {"train": [r["filename"] for r in csv.DictReader(f)]}
    for name in ("val", "test"):
        tokens = (split / f"{name}.txt").read_text().split()
        files[name] = [t for t in tokens if t.lower().endswith(IMAGE_EXTS)]
    val_a = _half({dog_of(f) for f in files["val"]}, ds["seed"])
    files["val_a"] = [f for f in files["val"] if dog_of(f) in val_a]
    files["val_b"] = [f for f in files["val"] if dog_of(f) not in val_a]
    return {s: [(str(images / f), dog_of(f)) for f in files[s]] for s in SPLITS}


def _dogfacenet(ds):
    """Seeded identity-disjoint 70/10/20 split. The 10% validation dogs are halved."""
    root = Path(ds["images"])
    photos = {d.name: sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
              for d in root.iterdir() if d.is_dir()}
    dogs = sorted(photos)
    random.Random(ds["seed"]).shuffle(dogs)
    a, b = round(0.7 * len(dogs)), round(0.8 * len(dogs))
    val_a = _half(dogs[a:b], ds["seed"])
    parts = {"train": dogs[:a], "val_a": [d for d in dogs[a:b] if d in val_a],
             "val_b": [d for d in dogs[a:b] if d not in val_a], "test": dogs[b:]}
    return {s: [(p, d) for d in sorted(parts[s]) for p in photos[d]] for s in SPLITS}


LOADERS = {"petface": _petface, "dogfacenet": _dogfacenet}


def check_disjoint(splits):
    """Leakage guard: no dog may appear in more than one split."""
    owner = {}
    for s in SPLITS:
        for d in {d for _, d in splits[s]}:
            if owner.setdefault(d, s) != s:
                raise ValueError(f"dog {d} is in both {owner[d]} and {s}")


def load_splits(cfg, freeze=False):
    """Build the splits, run the leakage guard, and check them against the frozen hash."""
    ds = cfg["dataset"]
    splits = LOADERS[ds["name"]](ds)
    check_disjoint(splits)
    sha = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    frozen = FROZEN_DIR / f"{ds['name']}.json"
    if freeze and not frozen.exists():
        FROZEN_DIR.mkdir(exist_ok=True)
        dogs = {s: sorted({d for _, d in splits[s]}) for s in SPLITS}
        frozen.write_text(json.dumps({"sha256": sha, "seed": ds["seed"], "dogs": dogs}, indent=1) + "\n")
    if not frozen.exists():
        sys.exit(f"{frozen} is missing; run make data")
    if json.loads(frozen.read_text())["sha256"] != sha:
        sys.exit(f"splits no longer match {frozen}; the data or the split code changed")
    return splits, sha


def gallery_query(items, seed, unknown_frac=0.0):
    """One random gallery photo per dog; every other photo is a query.

    With unknown_frac > 0 (open-set), that fraction of dogs loses its gallery photo.
    Gallery picks do not depend on unknown_frac, so the query set is the same in
    the closed-set and open-set variants. Returns (gallery, queries, unknown_dogs).
    """
    groups = by_dog(items)
    pick = random.Random(seed)
    gallery_of = {d: pick.choice(ix) for d, ix in groups.items()}
    unknown = set(random.Random(seed + 1).sample(list(groups), round(unknown_frac * len(groups))))
    gallery = [gallery_of[d] for d in groups if d not in unknown]
    queries = [i for d, ix in groups.items() for i in ix if i != gallery_of[d]]
    return gallery, queries, unknown


def verification_pairs(items, seed):
    """PetFace protocol (paper Sec. 4.1): per dog, one same-dog pair and one pair with
    a random photo of a different dog. Returns (i, j, label) index triples."""
    groups = {d: ix for d, ix in by_dog(items).items() if len(ix) >= 2}
    dogs, rng, pairs = list(groups), random.Random(seed), []
    for k, d in enumerate(dogs):
        a, b = rng.sample(groups[d], 2)
        j = rng.randrange(len(dogs) - 1)
        other = dogs[j + (j >= k)]
        pairs += [(a, b, 1), (a, rng.choice(groups[other]), 0)]
    return pairs


def official_pairs(cfg, items):
    """PetFace verification.csv as (i, j, label) triples into items; None for other datasets."""
    ds = cfg["dataset"]
    if "verification_csv" not in ds:
        return None
    index = {p: i for i, (p, _) in enumerate(items)}
    images = Path(ds["images"])
    with open(ds["verification_csv"], newline="") as f:
        return [(index[str(images / r["filename1"])], index[str(images / r["filename2"])], int(r["label"]))
                for r in csv.DictReader(f)]


if __name__ == "__main__":
    splits, sha = load_splits(load_config(sys.argv[1]), freeze=True)
    for s in SPLITS:
        print(f"{s}: {len({d for _, d in splits[s]})} dogs, {len(splits[s])} images")
    print("sha256", sha)
