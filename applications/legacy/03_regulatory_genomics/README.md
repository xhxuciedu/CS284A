# Regulatory motif scoring

**Question:** How can a transcription-factor binding profile distinguish motif-like DNA sequences from background sequence?

The script downloads the [JASPAR](https://jaspar.elixir.no/) CTCF position-frequency matrix `MA0139.1`, then generates **simulated** positive and GC-matched background sequences from that real motif. It scores sequences using a position-weight matrix and plots the score distribution. The simulated labels make the mathematics visible; they are not experimental binding measurements and do not show whether CTCF binds a site in a cell.

## Setup and run

```bash
cd applications/03_regulatory_genomics
./setup.sh
.venv/bin/python main.py
```

Windows: create `.venv` with `py -m venv .venv`, install `requirements.txt`, and run `main.py` with `.venv\\Scripts\\python`.

The script reports ROC AUC on simulated sequences. Change the pseudocount or GC background in the code, then examine how the score distribution changes.

**Discuss:** Why do negative-sequence design and genomic background composition matter when evaluating a motif model?
