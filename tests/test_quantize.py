"""The int8 detector must convert and run: frozen batch norm folded into the convs, and
the FPN's 1x1 pool left in fp32 because QNNPACK refuses it."""
import pytest

pytest.importorskip("torchvision")
from PIL import Image
from torchvision.transforms.functional import to_tensor

from pawmatch import headdet, quantize


def test_int8_detector_runs_with_a_quantized_backbone():
    photo = Image.new("RGB", (320, 240), (120, 90, 60))
    model = quantize.int8_detector(headdet.build(), [photo])
    out = model([to_tensor(photo)])[0]
    assert set(out) == {"boxes", "labels", "scores"}
    assert any(type(m).__module__.startswith("torch.ao.nn.quantized") for m in model.backbone.modules())
