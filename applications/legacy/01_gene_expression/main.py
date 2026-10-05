"""High-dimensional gene expression classification with leakage-safe feature selection."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def report(name, model, x_test, y_test):
    predicted = model.predict(x_test)
    print(f"\n{name}")
    print(f"balanced accuracy: {balanced_accuracy_score(y_test, predicted):.3f}")
    print(f"macro F1:          {f1_score(y_test, predicted, average='macro'):.3f}")
    print(classification_report(y_test, predicted, zero_division=0))
    return predicted


def main():
    # ucimlrepo cannot import dataset 401 ("not available for import"); use the shared cached loader instead
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # applications/course_utils.py
    from course_utils import load_tcga
    X, labels, genes = load_tcga()
    x = pd.DataFrame(X, columns=genes)
    y = pd.Series(labels)
    print(f"samples={len(x)}, features={x.shape[1]}, classes={y.nunique()}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    report("Majority-class baseline", baseline, x_test, y_test)

    model = make_pipeline(
        SelectKBest(score_func=f_classif, k=300),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    report("Selected genes + logistic regression", model, x_test, y_test)

    Path("outputs").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, xticks_rotation=45)
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
