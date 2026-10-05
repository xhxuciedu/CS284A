"""Volume-held-out random-forest baseline for an MSD segmentation task."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_volume(image_path, label_path):
    image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    label = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.int16)
    if image.shape != label.shape or image.ndim != 3:
        raise ValueError(f"Image/label shape mismatch: {image_path.name}")
    low, high = np.percentile(image, [1, 99])
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    return image, label


def features(image):
    grid = np.indices(image.shape, dtype=np.float32)
    coordinates = [grid[i] / max(image.shape[i] - 1, 1) for i in range(3)]
    return np.column_stack([image.ravel()] + [v.ravel() for v in coordinates])


def sampled_voxels(image, label, rng, per_class=3000):
    x = features(image)
    y = label.ravel()
    selected = []
    for value in np.unique(y):
        candidates = np.flatnonzero(y == value)
        selected.append(rng.choice(candidates, min(per_class, len(candidates)), replace=False))
    indices = np.concatenate(selected)
    return x[indices], y[indices]


def dice(truth, prediction, value):
    a, b = truth == value, prediction == value
    denominator = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denominator) if denominator else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/Task04_Hippocampus"))
    parser.add_argument("--train-volumes", type=int, default=5)
    args = parser.parse_args()
    image_dir, label_dir = args.data_dir / "imagesTr", args.data_dir / "labelsTr"
    pairs = [(p, label_dir / p.name) for p in sorted(image_dir.glob("*.nii.gz"))
             if (label_dir / p.name).exists()]
    if len(pairs) < args.train_volumes + 2:
        raise SystemExit(f"Need at least {args.train_volumes + 2} matched volumes in {args.data_dir}")
    train, test = train_test_split(pairs, test_size=2, random_state=42)
    train = train[:args.train_volumes]
    rng = np.random.default_rng(42)
    batches = [sampled_voxels(*load_volume(*pair), rng) for pair in train]
    x_train = np.concatenate([batch[0] for batch in batches])
    y_train = np.concatenate([batch[1] for batch in batches])
    model = RandomForestClassifier(
        n_estimators=60, max_depth=16, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    ).fit(x_train, y_train)
    Path("outputs").mkdir(exist_ok=True)
    for image_path, label_path in test:
        image, truth = load_volume(image_path, label_path)
        predicted = model.predict(features(image)).reshape(image.shape)
        scores = {int(value): dice(truth, predicted, value)
                  for value in np.unique(truth) if value != 0}
        print(f"{image_path.name}: foreground Dice {scores}")
        z = image.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, data, title in zip(axes, [image[:, :, z], truth[:, :, z], predicted[:, :, z]],
                                   ["MRI", "Ground truth", "Prediction"]):
            ax.imshow(data.T, origin="lower", cmap="gray" if title == "MRI" else "viridis")
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(f"outputs/{image_path.stem}_slice.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
