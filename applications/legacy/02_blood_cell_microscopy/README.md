# Blood-cell microscopy with a small CNN

**Question:** What can a convolutional network learn from microscope images of different blood cell types?

Data: [BloodMNIST in MedMNIST v2](https://medmnist.com/v2), a lightweight eight-class image benchmark. The package downloads the data on first run. Its 28 × 28 images are useful for explaining CNNs, but they cannot establish performance for real laboratory microscopy.

## Setup and run

```bash
cd applications/02_blood_cell_microscopy
./setup.sh
.venv/bin/python main.py --epochs 3 --train-limit 6000
```

Windows: create `.venv` with `py -m venv .venv`, install `requirements.txt`, and run `main.py` with `.venv\\Scripts\\python`.

The script trains only on the official training split, selects the best epoch with the validation split, and reports test macro F1 and a confusion matrix. The `--train-limit` setting controls CPU runtime.

**Discuss:** Which cell types are confused? What information might be missing at 28 × 28 pixels?
