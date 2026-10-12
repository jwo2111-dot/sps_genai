"""CNN image classifier for CIFAR-10, adapted from Module 4, Practical 3 (CNN)."""

import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGE_SIZE = 64
WEIGHTS_PATH = Path(__file__).parent / "weights" / "cnn_cifar10.pt"

# CIFAR-10 images are 32x32, so they are resized to the 64x64 input the architecture expects
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


class SimpleCNN(nn.Module):
    """Assignment 2 architecture: 64x64x3 input, two conv/pool blocks, FC 100, FC 10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 16 x 64 x 64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves height and width
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 32 x 32 x 32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16 x 32 x 32
        x = self.pool(F.relu(self.conv2(x)))  # 32 x 16 x 16
        x = torch.flatten(x, 1)  # 8192
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImageClassifier:
    """Loads trained SimpleCNN weights and classifies uploaded images."""

    def __init__(self, weights_path=WEIGHTS_PATH):
        self.model = SimpleCNN(num_classes=len(CLASSES))
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, image_bytes):
        """Return the predicted class and the probability of each class."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = transform(image).unsqueeze(0)  # add batch dimension: 1 x 3 x 64 x 64
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)[0]
        best = int(probs.argmax())
        return {
            "class": CLASSES[best],
            "confidence": float(probs[best]),
            "probabilities": {name: float(p) for name, p in zip(CLASSES, probs, strict=True)},
        }
