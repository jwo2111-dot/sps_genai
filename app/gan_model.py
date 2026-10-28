"""MNIST GAN (Assignment 3)."""

import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import make_grid

LATENT_DIM = 100
WEIGHTS_PATH = Path(__file__).parent / "weights" / "gan_generator_mnist.pt"


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 1 x 28 x 28
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 64 x 14 x 14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DigitGenerator:
    def __init__(self, weights_path=WEIGHTS_PATH):
        self.model = Generator()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def generate(self, count=16, seed=None):
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        z = torch.randn(count, LATENT_DIM, generator=generator)
        with torch.no_grad():
            images = self.model(z)
        return (images + 1) / 2  # [-1, 1] -> [0, 1]

    def generate_png(self, count=16, seed=None, scale=4):
        images = self.generate(count, seed)
        nrow = max(1, round(count**0.5))
        grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
        array = (grid[0] * 255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(array, mode="L")
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
