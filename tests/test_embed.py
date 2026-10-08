"""Each backbone must use its own normalization, not a shared one."""
import pytest

embed = pytest.importorskip("pawmatch.embed")

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
OPENAI_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


@pytest.mark.parametrize("name, mean_std, interpolation", [
    ("resnet50_in1k", IMAGENET, "bilinear"),
    ("clip_vitb16", OPENAI_CLIP, "bicubic"),
    ("dinov2_vits14", IMAGENET, "bicubic"),
])
def test_backbone_transform(name, mean_std, interpolation):
    _, tf = embed.build(name)
    resize, _, norm = tf.transforms
    assert resize.size == (224, 224) and resize.interpolation.value == interpolation
    assert norm.mean == pytest.approx(mean_std[0]) and norm.std == pytest.approx(mean_std[1])
