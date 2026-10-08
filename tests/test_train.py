import math

import pytest

train = pytest.importorskip("pawmatch.train")


def test_lr_schedule_warms_up_the_head_then_decays_both():
    # 20 steps: ceil(10%) = 2 head-only steps, then an 18-step cosine for both groups.
    scales = [train.lr_scales(s, 20) for s in range(20)]
    assert scales[:2] == [(1.0, 0.0), (1.0, 0.0)]
    assert scales[2] == (1.0, 1.0)
    assert scales[11] == pytest.approx((0.5, 0.5))  # halfway through the cosine
    assert scales[19][0] == pytest.approx(0.5 * (1 + math.cos(math.pi * 17 / 18)))
    assert train.lr_scales(1799, 18000) == (1.0, 0.0) and train.lr_scales(1800, 18000) == (1.0, 1.0)
