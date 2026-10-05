# Single-cell expression and cell populations

**Question:** What cell populations become visible after quality control and dimension reduction?

This tutorial uses the public [PBMC3k dataset in Scanpy](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html), a small teaching dataset, to make the pipeline inspectable. The lecture also introduces larger, more recent collections in [CZ CELLxGENE](https://cellxgene.cziscience.com/docs/01__CellxGene). The PBMC3k download happens on first run.

## Setup and run

```bash
cd applications/05_single_cell
./setup.sh
.venv/bin/python main.py
```

Windows: create `.venv` with `py -m venv .venv`, install `requirements.txt`, and run `main.py` with `.venv\\Scripts\\python`.

The code filters low-quality cells, normalizes counts, selects variable genes, computes PCA and UMAP, and uses k-means as a deliberately simple clustering baseline. The clusters are **not** verified cell types. It writes QC and UMAP plots to `outputs/`.

**Discuss:** How could donor effects or changes in preprocessing alter an apparent cluster? What evidence would support a biological cell-type label?
