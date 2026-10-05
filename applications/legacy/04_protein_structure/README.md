# Reading protein structure confidence

**Question:** Which parts of an AlphaFold prediction can be interpreted with more confidence?

The script queries the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/faq) for a UniProt accession and reads per-residue pLDDT from the prediction file. The default is `P69905` (human hemoglobin subunit alpha). pLDDT is a model confidence estimate, **not** a measurement of protein function or clinical effect.

## Setup and run

```bash
cd applications/04_protein_structure
./setup.sh
.venv/bin/python main.py --uniprot P69905
```

Windows: create `.venv` with `py -m venv .venv`, install `requirements.txt`, and run `main.py` with `.venv\\Scripts\\python`.

The script caches the PDB in `data/`, reports the confidence distribution, and draws pLDDT along the protein sequence. It does not train or evaluate a structure predictor. Experimental validation would require a suitable measured structure and careful alignment.

**Discuss:** Why can a high-confidence local fold still be insufficient evidence for a protein interaction or ligand-binding claim?
