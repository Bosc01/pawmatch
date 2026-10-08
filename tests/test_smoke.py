"""Smoke run on 500 DogFaceNet dogs: embed, evaluate, and two fine-tune steps."""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not Path("data/dogfacenet/after_4_bis").is_dir(),
                                reason="DogFaceNet is not downloaded")


def test_smoke_500_dogs():
    pytest.importorskip("torch")
    from pawmatch import data, embed, evaluate, train

    splits, _ = data.load_splits(data.load_config("configs/dogfacenet.toml"))
    every = [it for s in data.SPLITS for it in splits[s]]
    keep = set(sorted({d for _, d in every})[:500])
    items = [it for it in every if it[1] in keep]

    model, tf = embed.build("dinov2_vits14")
    emb = embed.encode(model, tf, [p for p, _ in items])
    assert emb.shape == (len(items), 384)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)

    res = evaluate.compute(items, emb, data.verification_pairs(items, 0), seed=0, unknown_frac=0.5, n_boot=20)
    assert res["closed_set"]["n_query_dogs"] == 500
    assert 0.1 < res["closed_set"]["rank1"]["value"] <= 1  # chance is 1/500
    assert 0.5 < res["verification"]["auc"]["value"] <= 1

    losses = train.fit(model, tf, items, hard_negative_weight=1.0, epochs=1, max_steps=2)
    assert len(losses) == 2 and all(np.isfinite(list(step.values())).all() for step in losses)
