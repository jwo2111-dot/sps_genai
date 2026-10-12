import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.cnn_model import CLASSES, SimpleCNN
from app.main import app

client = TestClient(app)


def image_bytes(size=(32, 32), color=(200, 30, 30), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def test_architecture_matches_spec():
    model = SimpleCNN()
    x = torch.zeros(4, 3, 64, 64)

    x = model.pool(torch.relu(model.conv1(x)))
    assert x.shape == (4, 16, 32, 32)
    x = model.pool(torch.relu(model.conv2(x)))
    assert x.shape == (4, 32, 16, 16)
    assert model.fc1.in_features == 32 * 16 * 16
    assert model.fc1.out_features == 100
    assert model.fc2.out_features == 10
    assert model(torch.zeros(4, 3, 64, 64)).shape == (4, 10)


def test_layer_hyperparameters():
    model = SimpleCNN()
    for conv, filters in ((model.conv1, 16), (model.conv2, 32)):
        assert conv.out_channels == filters
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (1, 1)
        assert conv.padding == (1, 1)
    assert model.pool.kernel_size == 2
    assert model.pool.stride == 2


def test_classify_endpoint():
    files = {"file": ("red.png", image_bytes(), "image/png")}
    response = client.post("/classify", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["class"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert body["confidence"] == max(body["probabilities"].values())


def test_classify_accepts_any_size_and_format():
    files = {"file": ("photo.jpg", image_bytes(size=(640, 480), fmt="JPEG"), "image/jpeg")}
    assert client.post("/classify", files=files).status_code == 200


def test_classify_rejects_non_image():
    files = {"file": ("notes.txt", b"not an image", "text/plain")}
    assert client.post("/classify", files=files).status_code == 400


def test_classes_endpoint():
    assert client.get("/classes").json()["classes"] == CLASSES
