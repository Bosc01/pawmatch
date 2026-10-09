import numpy as np
import pytest

xattn = pytest.importorskip("pawmatch.xattn")


def test_hard_negatives_are_the_nearest_other_dogs():
    emb = np.array([[1, 0], [0.99, 0.14], [0.9, 0.44], [0, 1]], np.float32)
    dogs = np.array(["a", "a", "b", "c"])
    neg = xattn.hard_negatives(emb, dogs, 2)
    assert neg[0].tolist() == [2, 3]  # photo 1 is the same dog, so it is skipped
    assert neg[3].tolist() == [2, 1]


def test_head_scores_one_logit_per_pair():
    torch = pytest.importorskip("torch")
    head = xattn.CrossAttentionHead()
    assert head(torch.randn(5, 256, 384), torch.randn(5, 256, 384)).shape == (5,)
