# Medical image segmentation

**Question:** How can an anatomy mask be evaluated on a complete, unseen 3D volume?

Download the **Hippocampus** task from the [Medical Segmentation Decathlon](https://medicaldecathlon.com/) and extract it under `data/Task04_Hippocampus/` so `imagesTr/` and `labelsTr/` are present. The tutorial trains a deliberately simple random-forest voxel baseline on a few volumes, then measures Dice overlap on held-out volumes. It is a teaching baseline, not a competitive segmentation system. The dataset is a larger manual download; review its size before downloading.

## Setup and run

```bash
cd applications/06_medical_image_segmentation
./setup.sh
.venv/bin/python main.py --data-dir data/Task04_Hippocampus --train-volumes 5
```

Windows: create `.venv` with `py -m venv .venv`, install `requirements.txt`, and run `main.py` with `.venv\\Scripts\\python`.

Volumes are split **before** voxel sampling, so neighboring slices from one person cannot enter both train and test sets. The script reports Dice for foreground labels and saves one held-out slice with ground truth and prediction.

**Discuss:** Why can a model obtain a high background-pixel accuracy while missing the anatomical structure? How does Dice differ from boundary quality?
