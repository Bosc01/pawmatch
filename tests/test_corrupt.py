import numpy as np
import pytest
from PIL import Image

corrupt = pytest.importorskip("pawmatch.corrupt")

IMG = Image.fromarray(np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", corrupt.CORRUPTIONS)
def test_corruption_is_deterministic_and_changes_the_image(name):
    a = corrupt.corrupt(IMG, name, np.random.default_rng(1))
    b = corrupt.corrupt(IMG, name, np.random.default_rng(1))
    assert a.size == IMG.size and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(IMG))
