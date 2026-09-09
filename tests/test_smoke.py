import io

import numpy as np
import pytest
import torch
from PIL import Image

from model.Meso4.model import build_model as build_meso4
from model.SwinTransformer.model import build_model as build_swin


@pytest.fixture(scope="module")
def sample_image_bytes():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(300, 280, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_meso4_output_shape():
    model = build_meso4().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_swin_output_shape():
    model = build_swin(num_classes=1, weights=None).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1)


def test_meso4_param_count():
    n = sum(p.numel() for p in build_meso4().parameters())
    assert 20_000 < n < 40_000


def test_preprocess_matches_training_transform(sample_image_bytes):
    from torchvision import transforms

    from api.app import _eval_transform

    img = Image.open(io.BytesIO(sample_image_bytes)).convert("RGB")
    ours = _eval_transform(img)
    ref = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(img)
    assert torch.equal(ours, ref)


@pytest.mark.skipif(
    not __import__("os").path.exists("models/meso4/model.onnx"),
    reason="exported ONNX artifact not present",
)
def test_onnx_inference_roundtrip(sample_image_bytes):
    import onnxruntime as ort

    from api.app import predict, preprocess

    session = ort.InferenceSession(
        "models/meso4/model.onnx", providers=["CPUExecutionProvider"])
    tensor = preprocess(sample_image_bytes)
    logits = session.run(["logits"], {session.get_inputs()[0].name: tensor})
    assert logits[0].shape == (1, 1)

    result = predict(sample_image_bytes)
    assert set(result) == {"label", "confidence"}
    assert result["label"] in {"Real", "Fake"}
    assert 0 <= result["confidence"] <= 100


def test_data_loader_splits_consistent(tmp_path):

    from data.data_loader import get_data_loaders

    rng = np.random.default_rng(1)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(2):
                Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
                    d / f"{i}.jpg")

    _, _, _, meta = get_data_loaders(str(tmp_path), batch_size=2, num_workers=0)
    assert meta["class_to_idx"] == {"fake": 0, "real": 1}
    assert meta["sizes"] == {"train": 4, "val": 4, "test": 4}
