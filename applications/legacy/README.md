# Legacy application scripts (superseded by the companion notebooks)

These stand-alone scripts predate the lecture-by-lecture notebooks in `applications/Lxx_*/` and are kept for reference.

Each folder is an independent Python tutorial. Run `./setup.sh` **inside that folder** to create its local `.venv`; then use `.venv/bin/python main.py` (macOS/Linux) or `.venv\\Scripts\\python.exe main.py` (Windows, after creating a venv manually). The code downloads public data only when its README says so. Generated data and figures go in `data/` and `outputs/`, which Git ignores.

| Folder | Main question | Data source |
| --- | --- | --- |
| `01_gene_expression` | What patterns distinguish tumor tissue types? | UCI TCGA PANCAN RNA-seq teaching extract |
| `02_blood_cell_microscopy` | How well can a CNN distinguish blood cell images? | BloodMNIST |
| `03_regulatory_genomics` | How does a sequence motif score candidate binding sites? | JASPAR motif profile and simulated sequences |
| `04_protein_structure` | Which regions of a predicted structure have high confidence? | AlphaFold Protein Structure Database |
| `05_single_cell` | What populations appear in a single-cell expression matrix? | Scanpy PBMC3k teaching dataset |
| `06_medical_image_segmentation` | How are 3D segmentation results evaluated? | Medical Segmentation Decathlon, Hippocampus task |
| `07_digital_pathology` | How sensitive is a patch classifier to color shift? | PathMNIST; simulated stain shift |
| `08_protein_mutation_effects` | Can sequence features rank experimental mutation effects? | ProteinGym DMS assay |

The small image sets and simulated shifts are teaching devices. Their scores must not be interpreted as clinical performance.
