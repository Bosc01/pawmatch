"""The int8 detector must convert and run: frozen batch norm folded into the convs, and
the FPN's 1x1 pool left in fp32 because QNNPACK refuses it."""
import pytest

pytest.importorskip("torchvision")
from PIL import Image
from torchvision.transforms.functional import to_tensor

from pawmatch import detect, headdet, quantize


def test_calibration_photos_are_oxford_half_a_training_only():
    if not detect.OXFORD.exists():
        pytest.skip("no Oxford-IIIT Pet data")
    dogs, train_idx, val_idx = headdet.half_a_split()
    _, _, halves = detect.oxford_halves()
    photos = set(quantize.calibration_photos())
    assert len(photos) == quantize.CALIBRATION and photos <= {dogs[i][0] for i in train_idx}
    assert not photos & {dogs[i][0] for i in halves["B"] + val_idx}


def test_int8_detector_runs_with_a_quantized_backbone():
    photo = Image.new("RGB", (320, 240), (120, 90, 60))
    model = quantize.int8_detector(headdet.build(), [photo])
    out = model([to_tensor(photo)])[0]
    assert set(out) == {"boxes", "labels", "scores"}
    assert any(type(m).__module__.startswith("torch.ao.nn.quantized") for m in model.backbone.modules())
