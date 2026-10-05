# Protein mutation effects

**Question:** Can a simple sequence-feature baseline rank the measured effects of single amino-acid substitutions?

Data: one experimental deep-mutational-scanning assay from the [ProteinGym](https://proteingym.org/) substitutions benchmark. `prepare_data.py` downloads the official cross-validation-fold archive (about 50 MB) into `data/`; `main.py` chooses one compatible assay CSV by default. You can pass `--csv path/to/assay.csv` to choose another. Only the experimental assay scores are used, not the clinical benchmark.

## Setup and run

```bash
cd applications/08_protein_mutation_effects
./setup.sh
.venv/bin/python prepare_data.py
.venv/bin/python main.py
```

Windows: create `.venv` with `py -m venv .venv`, install `requirements.txt`, then run both scripts with `.venv\\Scripts\\python`.

The code keeps only single substitutions, encodes the original amino acid, position, and substituted amino acid, then compares a mean baseline with ridge regression. It reports mean absolute error and Spearman rank correlation under a random variant split and a stricter held-out-position split. A limited score on unseen positions is expected from this deliberately simple representation.

**Discuss:** Why is a high score on randomly held-out variants weak evidence about unseen proteins or families?
