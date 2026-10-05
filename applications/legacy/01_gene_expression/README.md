# Gene expression and tissue type

**Question:** Can a simple model distinguish five tumor tissue types from RNA-seq expression measurements?

Data: [UCI Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene), a teaching extract of TCGA PANCAN with 801 samples and 20,531 anonymous features. The shared loader `course_utils.load_tcga()` downloads about 70 MB on first use and caches it in `applications/data/`. The anonymous gene names prevent biological interpretation of individual features; the lesson is about high-dimensional modeling and evaluation. This is tissue classification, not patient outcome prediction.

## Setup and run

```bash
cd applications/01_gene_expression
./setup.sh
.venv/bin/python main.py
```

Windows: `py -m venv .venv`, `.venv\\Scripts\\python -m pip install -r requirements.txt`, then `.venv\\Scripts\\python main.py`.

The script compares a majority-class baseline with a feature-selection and logistic-regression pipeline. Feature selection runs **inside** the fitted pipeline after the train/test split. It prints balanced accuracy and macro F1 and writes a confusion matrix to `outputs/`.

**Discuss:** Why is a random sample split adequate for this limited exercise but insufficient evidence that a classifier will generalize to another sequencing site or cohort?
