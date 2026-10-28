"""Train the MNIST GAN: uv run python train_gan.py [--epochs 20]"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from app.gan_model import LATENT_DIM, WEIGHTS_PATH, Discriminator, Generator
from train_cnn import get_device

BATCH_SIZE = 128


def weights_init(module):
    # DCGAN-style init
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--samples-dir", default="gan_progress")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # [-1, 1] to match tanh
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM, device=device)
    real_label = torch.ones(BATCH_SIZE, 1, device=device)
    fake_label = torch.zeros(BATCH_SIZE, 1, device=device)
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        total_d, total_g, d_real_mean, d_fake_mean = 0.0, 0.0, 0.0, 0.0
        progress = tqdm(loader, ncols=100, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for real, _ in progress:
            real = real.to(device)
            fake = generator(torch.randn(BATCH_SIZE, LATENT_DIM, device=device))

            # train D
            opt_d.zero_grad()
            d_real = discriminator(real)
            d_fake = discriminator(fake.detach())
            loss_d = criterion(d_real, real_label) + criterion(d_fake, fake_label)
            loss_d.backward()
            opt_d.step()

            # train G
            opt_g.zero_grad()
            loss_g = criterion(discriminator(fake), real_label)
            loss_g.backward()
            opt_g.step()

            total_d += loss_d.item()
            total_g += loss_g.item()
            d_real_mean += d_real.mean().item()
            d_fake_mean += d_fake.mean().item()

        n = len(loader)
        print(
            f"Epoch {epoch + 1}: D loss {total_d / n:.3f}, G loss {total_g / n:.3f}, "
            f"D(real) {d_real_mean / n:.2f}, D(fake) {d_fake_mean / n:.2f}"
        )

        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise)
        save_image((samples + 1) / 2, samples_dir / f"epoch_{epoch + 1:02d}.png", nrow=8)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), WEIGHTS_PATH)
    print(f"Generator weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
