"""A transparent baseline for one ProteinGym deep-mutational-scanning assay."""
import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SINGLE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def compatible_csv(path):
    try:
        columns = pd.read_csv(path, nrows=1).columns
    except Exception:
        return False
    return {"mutant", "DMS_score"}.issubset(columns)


def find_csv(explicit):
    if explicit:
        return explicit
    options = [p for p in Path("data").rglob("*.csv") if compatible_csv(p)]
    if not options:
        raise SystemExit("No assay CSV found. Run prepare_data.py or pass --csv.")
    return sorted(options, key=lambda p: p.stat().st_size)[0]


def load_assay(path):
    data = pd.read_csv(path)
    data["DMS_score"] = pd.to_numeric(data["DMS_score"], errors="coerce")
    data = data.dropna(subset=["mutant", "DMS_score"]).copy()
    data["match"] = data["mutant"].astype(str).map(SINGLE.fullmatch)
    data = data[data["match"].notna()].copy()
    if len(data) < 100:
        raise ValueError(f"Too few single substitutions in {path}")
    matches = data["match"]
    features = [{"wild_type": m[1], "position": m[2], "new_amino_acid": m[3]}
                for m in matches]
    groups = np.asarray([int(m[2]) for m in matches])
    return features, data["DMS_score"].to_numpy(), groups


def evaluate_split(name, train_idx, test_idx, features, y):
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform([features[i] for i in train_idx])
    x_test = vectorizer.transform([features[i] for i in test_idx])
    results = {}
    for label, model in [("mean", DummyRegressor()), ("ridge", Ridge(alpha=10.0))]:
        model.fit(x_train, y[train_idx])
        predicted = model.predict(x_test)
        correlation = spearmanr(y[test_idx], predicted).statistic
        results[label] = (mean_absolute_error(y[test_idx], predicted), correlation)
    print(f"{name} (n={len(test_idx)}):")
    for label, (mae, rho) in results.items():
        print(f"  {label:5s} MAE={mae:.3f}; Spearman rho={rho:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    path = find_csv(args.csv)
    features, y, positions = load_assay(path)
    indices = np.arange(len(y))
    random_train, random_test = train_test_split(indices, test_size=0.25, random_state=42)
    group_train, group_test = next(GroupShuffleSplit(test_size=0.25, random_state=42).split(
        indices, y, groups=positions))
    print(f"assay={path.name}, single substitutions={len(y)}, positions={len(set(positions))}")
    random_result = evaluate_split("Random variant split", random_train, random_test, features, y)
    group_result = evaluate_split("Held-out position split", group_train, group_test, features, y)
    Path("outputs").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Random variants", "Unseen positions"]
    values = [random_result["ridge"][0], group_result["ridge"][0]]
    ax.bar(names, values)
    ax.set(ylabel="Mean absolute error", title="Split choice changes the question")
    fig.tight_layout()
    fig.savefig("outputs/split_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
