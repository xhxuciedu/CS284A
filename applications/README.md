# Companion notebooks

One folder per lecture application. Each folder holds the notebook source (jupytext percent format, the file to edit)
and the executed notebook:

```
applications/
├── course_utils.py          shared helpers (seeds, plot style, cached downloads, TCGA/JASPAR loaders, logos)
├── make_notebooks.py        convert + execute sources → .ipynb (logs runtimes to notebook_runs.tsv)
├── requirements.txt         one environment for every notebook
├── data/                    downloaded datasets and cached results (created on first run, ignored by git)
├── L01_data_exploration/
│   ├── L01_data_exploration.py      ← source (edit this)
│   └── L01_data_exploration.ipynb   ← executed notebook (open this)
├── L02_knn_gene_expression/
├── …
├── L20_alphafold_confidence/
└── legacy/                  earlier stand-alone scripts (01_… to 08_…), superseded by the notebooks
```

See `../NOTEBOOK_INDEX.md` for the list of notebooks, datasets, runtimes and GPU needs.

## Setup (once)

```bash
python3 -m venv .venv                                  # at the repository root
.venv/bin/pip install -r applications/requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name python3
```

Windows: use `.venv\Scripts\pip` and `.venv\Scripts\python`.

## Running

- Interactively: open `applications/Lxx_…/Lxx_….ipynb` in Jupyter / VS Code with the `.venv` kernel. The notebooks
  import `course_utils` from the parent folder, so run them from their own folder (Jupyter's default).
- Rebuild from source: `cd applications && ../.venv/bin/python make_notebooks.py L08` (prefix match; `all` for every
  notebook; `--no-exec` to convert only; `--cpu` to hide GPUs when measuring CPU runtimes).

Data are public research or benchmark data, downloaded on first use into `applications/data/`. The small image sets
and simulated shifts are teaching devices; no score here is a clinical performance claim.
