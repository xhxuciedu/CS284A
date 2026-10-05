"""Shared helpers for the CS184A/284A companion notebooks.

Keeps notebooks short: reproducible seeds, a consistent plot style, and cached
downloads into applications/data/ (ignored by git).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# torch + numba (scanpy, UMAP) can segfault with the default threading layer on macOS
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

PALETTE = ["#0064A4", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def seed_everything(seed: int = 0):
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def plot_style():
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "figure.figsize": (6, 4), "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=PALETTE), "axes.titleweight": "bold", "font.size": 11,
        "legend.frameon": False, "image.cmap": "viridis",
    })


def download(url: str, name: str | None = None, timeout: int = 120) -> Path:
    """Download url to applications/data/<name> once; return the local path."""
    import requests
    name = name or url.rstrip("/").split("/")[-1]
    path = DATA / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.rename(path)
    return path


def device():
    """Best available torch device (CUDA > Apple MPS > CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


TCGA_URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"


def load_tcga():
    """UCI 'Gene expression cancer RNA-Seq' (TCGA PANCAN HiSeq extract), cached as a compressed .npz.

    Returns X (801 x 20531 float32, values as distributed), y (tumor-type strings), gene names (anonymized gene_0...).
    Tumor types: BRCA breast, COAD colon, KIRC kidney, LUAD lung, PRAD prostate.
    """
    import io
    import tarfile
    import zipfile
    import pandas as pd
    cache = DATA / "tcga_uci401_course.npz"
    if not cache.exists():
        z = download(TCGA_URL, "tcga_pancan_uci401.zip")
        with zipfile.ZipFile(z) as zf:
            inner = [n for n in zf.namelist() if n.endswith(".tar.gz")][0]
            with tarfile.open(fileobj=io.BytesIO(zf.read(inner))) as tf:
                data = pd.read_csv(tf.extractfile([m for m in tf.getmembers() if m.name.endswith("data.csv")][0]),
                                   index_col=0)
                labels = pd.read_csv(tf.extractfile([m for m in tf.getmembers() if m.name.endswith("labels.csv")][0]),
                                     index_col=0)
        labels = labels.loc[data.index]
        np.savez_compressed(cache, X=data.values.astype(np.float32), y=np.array(labels.iloc[:, 0].tolist(), dtype="U8"),
                            genes=np.array(list(data.columns), dtype="U16"), samples=np.array(list(data.index), dtype="U16"))
    f = np.load(cache, allow_pickle=False)
    return f["X"], f["y"], f["genes"]


# ---------------------------------------------------------------------------
# DNA motifs (L03, L11, L12)
# ---------------------------------------------------------------------------
BASES = "ACGT"
BASE_COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}


def load_jaspar(matrix_id: str = "MA0139.1"):
    """Position frequency matrix from JASPAR (https://jaspar.elixir.no), cached. Returns (name, counts W x 4 in ACGT order)."""
    import json
    path = DATA / f"jaspar_{matrix_id}.json"
    if not path.exists():
        download(f"https://jaspar.elixir.no/api/v1/matrix/{matrix_id}/?format=json", path.name)
    js = json.loads(path.read_text())
    counts = np.array([js["pfm"][b] for b in BASES], dtype=float).T
    return js["name"], counts


def pwm_from_counts(counts, pseudocount=0.5):
    """Position probability matrix Θ (W x 4): θ_{w,a} = p(s_w = a | motif), with pseudocounts."""
    c = counts + pseudocount
    return c / c.sum(1, keepdims=True)


def information_content(theta, background=None):
    q = np.full(4, 0.25) if background is None else np.asarray(background)
    return (theta * np.log2(theta / q)).sum(1)


def plot_logo(ax, theta, background=None, title=None):
    """Sequence logo: letter heights = θ_{w,a} × information content (bits) at position w."""
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
    import matplotlib.transforms as mt
    ic = information_content(theta, background)
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    for w in range(theta.shape[0]):
        y = 0.0
        for a in np.argsort(theta[w]):
            h = theta[w, a] * ic[w]
            if h < 1e-3:
                continue
            letter = BASES[a]
            tp = TextPath((0, 0), letter, size=1, prop=fp)
            bb = tp.get_extents()
            tr = (mt.Affine2D().translate(-bb.x0, -bb.y0).scale(0.9 / bb.width, h / bb.height)
                  .translate(w + 0.05 + 1, y))
            ax.add_patch(PathPatch(tr.transform_path(tp), color=BASE_COLORS[letter], lw=0))
            y += h
    ax.set_xlim(0.9, theta.shape[0] + 1.1)
    ax.set_ylim(0, 2)
    ax.set_ylabel("bits")
    ax.set_xticks(np.arange(1, theta.shape[0] + 1) + 0.5, np.arange(1, theta.shape[0] + 1))
    if title:
        ax.set_title(title)
    return ax


def sample_from_pwm(theta, rng, n):
    """Draw n motif instances (strings) from a position probability matrix."""
    W = theta.shape[0]
    u = rng.random((n, W, 1))
    idx = (u > np.cumsum(theta, 1)[None]).sum(-1)
    return ["".join(BASES[i] for i in row) for row in idx]


def random_dna(rng, n, L, gc=0.5):
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    idx = rng.choice(4, size=(n, L), p=p)
    return ["".join(BASES[i] for i in row) for row in idx]
