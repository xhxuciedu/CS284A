# Digital pathology and image shift

**Question:** How sensitive is a tissue-patch classifier to a change in image color?

This runnable exercise uses [PathMNIST](https://medmnist.com/v2), a small colon-pathology image benchmark. The script trains a PCA plus logistic-regression baseline, evaluates the official test split, then applies a **simulated color shift** to the same test images. The color operation illustrates sensitivity; it is not a measured laboratory or stain shift.

The accompanying lecture uses [CAMELYON17](https://camelyon17.grand-challenge.org/Data/) to explain real whole-slide images and multi-center evaluation. CAMELYON17 is too large for a default beginner tutorial, so this code does not download it.

## Setup and run

```bash
cd applications/07_digital_pathology
./setup.sh
.venv/bin/python main.py --train-limit 5000
```

Windows: create `.venv` with `py -m venv .venv`, install `requirements.txt`, and run `main.py` with `.venv\\Scripts\\python`.

The script reports macro F1 before and after the shift and saves example images and confusion matrices in `outputs/`.

**Discuss:** What metadata and split design would be needed to test a claim of generalization to a new pathology laboratory?
