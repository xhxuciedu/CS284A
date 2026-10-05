"""A compact CNN for the BloodMNIST teaching benchmark."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor


class SmallCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            truth.extend(labels.view(-1).numpy().tolist())
            predicted.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.asarray(truth), np.asarray(predicted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=6000)
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    info = medmnist.INFO["bloodmnist"]
    dataset_type = getattr(medmnist, info["python_class"])
    datasets = {split: dataset_type(split=split, download=True, transform=ToTensor())
                for split in ("train", "val", "test")}
    rng = np.random.default_rng(42)
    indices = rng.choice(len(datasets["train"]),
                         size=min(args.train_limit, len(datasets["train"])), replace=False)
    train = Subset(datasets["train"], indices.tolist())
    loaders = {
        "train": DataLoader(train, batch_size=128, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=256),
        "test": DataLoader(datasets["test"], batch_size=256),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(classes=len(info["label"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_score, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.view(-1).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
        y_val, p_val = evaluate(model, loaders["val"], device)
        score = f1_score(y_val, p_val, average="macro")
        print(f"epoch {epoch + 1}: validation macro F1={score:.3f}")
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    y_test, p_test = evaluate(model, loaders["test"], device)
    print(classification_report(y_test, p_test, zero_division=0))
    Path("outputs").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, p_test, ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
