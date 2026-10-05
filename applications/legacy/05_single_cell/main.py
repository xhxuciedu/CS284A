"""A compact single-cell workflow using PBMC3k and simple clustering."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    np.random.seed(42)
    sc.settings.verbosity = 1
    adata = sc.datasets.pbmc3k()
    print(f"raw matrix: {adata.n_obs} cells x {adata.n_vars} genes")
    adata.var["mitochondrial"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mitochondrial"], percent_top=None, inplace=True
    )
    Path("outputs").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(adata.obs["n_genes_by_counts"], bins=40)
    axes[0].set(xlabel="Detected genes per cell", ylabel="Cells")
    axes[1].hist(adata.obs["pct_counts_mitochondrial"], bins=40)
    axes[1].set(xlabel="Mitochondrial percentage", ylabel="Cells")
    fig.tight_layout()
    fig.savefig("outputs/qc.png", dpi=160)
    plt.close(fig)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs["pct_counts_mitochondrial"] < 20].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
    sc.tl.umap(adata, random_state=42)
    x_pca = adata.obsm["X_pca"][:, :20]
    labels = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(x_pca)
    adata.obs["cluster"] = labels.astype(str)
    score = silhouette_score(x_pca, labels, sample_size=min(2000, len(labels)), random_state=42)
    print(f"retained: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"PCA-space silhouette for 8 clusters: {score:.3f}")
    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(adata, color="cluster", ax=ax, show=False)
    fig.tight_layout()
    fig.savefig("outputs/umap_clusters.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
