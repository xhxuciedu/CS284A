"""PathMNIST baseline with a clearly labeled synthetic color-shift probe."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prepare(images):
    return images.astype(np.float32).reshape(len(images), -1) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=2000)
    args = parser.parse_args()
    info = medmnist.INFO["pathmnist"]
    kind = getattr(medmnist, info["python_class"])
    train = kind(split="train", download=True)
    test = kind(split="test", download=True)
    rng = np.random.default_rng(42)
    tr = rng.choice(len(train), min(args.train_limit, len(train)), replace=False)
    te = rng.choice(len(test), min(args.test_limit, len(test)), replace=False)
    x_train, y_train = prepare(train.imgs[tr]), train.labels[tr].ravel()
    original = test.imgs[te]
    x_test, y_test = prepare(original), test.labels[te].ravel()
    model = make_pipeline(
        StandardScaler(), PCA(n_components=80, random_state=42),
        LogisticRegression(max_iter=1000),
    ).fit(x_train, y_train)
    normal_prediction = model.predict(x_test)
    shifted = np.clip(original.astype(np.float32) * [1.15, 0.85, 1.0], 0, 255).astype(np.uint8)
    shifted_prediction = model.predict(prepare(shifted))
    normal_f1 = f1_score(y_test, normal_prediction, average="macro")
    shifted_f1 = f1_score(y_test, shifted_prediction, average="macro")
    print(f"original test macro F1: {normal_f1:.3f}")
    print(f"simulated color-shift macro F1: {shifted_f1:.3f}")
    Path("outputs").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    axes[0].imshow(original[0]); axes[0].set_title("Original patch")
    axes[1].imshow(shifted[0]); axes[1].set_title("Simulated shift")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("outputs/color_shift.png", dpi=160)
    plt.close(fig)
    for name, prediction in [("original", normal_prediction), ("shifted", shifted_prediction)]:
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, prediction, ax=ax, colorbar=False)
        fig.tight_layout()
        fig.savefig(f"outputs/confusion_{name}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
