"""Train SimpleCNN on CIFAR-10 and save the weights used by the API.

Usage: uv run python train_cnn.py [--epochs 10]
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.cnn_model import IMAGE_SIZE, WEIGHTS_PATH, SimpleCNN, transform

BATCH_SIZE = 32


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = model(inputs).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Random flips help the model generalize; the test set uses the plain transform
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        progress = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        train_accuracy = running_correct / running_total
        test_accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch + 1}: loss {running_loss / running_total:.4f}, "
            f"train acc {train_accuracy:.3f}, test acc {test_accuracy:.3f}"
        )

        # Keep the weights from the epoch with the best test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_PATH)

    print(f"Best test accuracy: {best_accuracy:.3f}. Weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
