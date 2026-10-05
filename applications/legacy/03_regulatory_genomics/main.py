"""Position-weight-matrix scoring using a real JASPAR motif and simulated sequences."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.metrics import RocCurveDisplay, roc_auc_score

MOTIF_URL = "https://jaspar.elixir.no/api/v1/matrix/MA0139.1/?format=json"
BASES = "ACGT"


def get_motif():
    response = requests.get(MOTIF_URL, timeout=30)
    response.raise_for_status()
    pfm = response.json()["pfm"]
    counts = np.asarray([pfm[base] for base in BASES], dtype=float)
    if counts.shape[0] != 4 or counts.shape[1] < 4:
        raise ValueError("Unexpected JASPAR motif matrix")
    return counts


def draw_sequences(probabilities, count, rng):
    positions = np.arange(probabilities.shape[1])
    return np.stack([rng.choice(4, p=probabilities[:, pos], size=count)
                     for pos in positions], axis=1)


def score_sequences(sequences, log_odds):
    return log_odds[sequences, np.arange(sequences.shape[1])].sum(axis=1)


def main():
    rng = np.random.default_rng(42)
    counts = get_motif()
    background = np.array([0.30, 0.20, 0.20, 0.30])  # A,C,G,T
    pseudocount = 0.5
    pwm = (counts + pseudocount) / (counts.sum(axis=0) + 4 * pseudocount)
    log_odds = np.log2(pwm / background[:, None])
    n, length = 1500, counts.shape[1]
    positive = draw_sequences(pwm, n, rng)
    negative = rng.choice(4, size=(n, length), p=background)
    sequences = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    scores = score_sequences(sequences, log_odds)
    print(f"JASPAR motif length: {length} bases")
    print(f"Simulated ROC AUC: {roc_auc_score(labels, scores):.3f}")
    print("Example sequence:", "".join(BASES[i] for i in sequences[0]))

    Path("outputs").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(scores[labels == 0], bins=35, alpha=0.7, label="GC-matched background")
    axes[0].hist(scores[labels == 1], bins=35, alpha=0.7, label="Motif-generated")
    axes[0].set(xlabel="PWM log-odds score", ylabel="Sequence count")
    axes[0].legend()
    RocCurveDisplay.from_predictions(labels, scores, ax=axes[1])
    fig.tight_layout()
    fig.savefig("outputs/motif_scores.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
