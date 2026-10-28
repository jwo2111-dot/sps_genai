import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.gan_model import LATENT_DIM, Discriminator, Generator
from app.main import app

client = TestClient(app)


def test_generator_architecture():
    g = Generator()
    x = g.fc(torch.randn(4, LATENT_DIM)).view(-1, 128, 7, 7)
    x = g.net[2](g.net[1](g.net[0](x)))
    assert x.shape == (4, 64, 14, 14)
    out = g(torch.randn(4, LATENT_DIM))
    assert out.shape == (4, 1, 28, 28)
    assert out.min() >= -1 and out.max() <= 1


def test_generator_layers():
    g = Generator()
    assert g.fc.in_features == 100 and g.fc.out_features == 7 * 7 * 128
    for layer, (c_in, c_out) in ((g.net[0], (128, 64)), (g.net[3], (64, 1))):
        assert (layer.in_channels, layer.out_channels) == (c_in, c_out)
        assert layer.kernel_size == (4, 4) and layer.stride == (2, 2) and layer.padding == (1, 1)
    assert isinstance(g.net[1], torch.nn.BatchNorm2d)
    assert isinstance(g.net[2], torch.nn.ReLU)
    assert isinstance(g.net[4], torch.nn.Tanh)


def test_discriminator_architecture():
    d = Discriminator()
    x = torch.randn(4, 1, 28, 28)
    assert d.net[1](d.net[0](x)).shape == (4, 64, 14, 14)
    assert d.net[0].kernel_size == (4, 4) and d.net[0].stride == (2, 2)
    assert d.net[2].out_channels == 128
    assert d.net[1].negative_slope == 0.2 and d.net[4].negative_slope == 0.2
    out = d(x)
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_digits_endpoint_returns_png_grid():
    response = client.get("/gan/digits", params={"count": 16, "seed": 1})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(response.content))
    assert image.size == (4 * (4 * 28 + 5 * 2), 4 * (4 * 28 + 5 * 2))


def test_digits_seed_is_reproducible():
    first = client.get("/gan/digits", params={"count": 4, "seed": 7}).content
    second = client.get("/gan/digits", params={"count": 4, "seed": 7}).content
    other = client.get("/gan/digits", params={"count": 4, "seed": 8}).content
    assert first == second
    assert first != other


def test_digits_count_limits():
    assert client.get("/gan/digits", params={"count": 0}).status_code == 422
    assert client.get("/gan/digits", params={"count": 65}).status_code == 422
