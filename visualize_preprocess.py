"""Before/after figure + histograms for the report (Section 5.1).

python visualize_preprocess.py --config config_exp2.yaml --data_root /content/Synapse --out runs/exp2/preprocess_examples.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import yaml

from preprocess import build_preprocess


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_list(list_dir, name):
    with open(os.path.join(list_dir, name)) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_exp2.yaml")
    ap.add_argument("--data_root", default="C:/Users/User/Downloads/cv_dataset")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="preprocess_examples.png")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = a.data_root or cfg["data_root"]
    pre = build_preprocess(cfg)
    names = read_list(cfg["list_dir"], "train.txt")
    pick = names[:: max(1, len(names) // a.n)][: a.n]
    fig, ax = plt.subplots(a.n, 4, figsize=(14, 3.6 * a.n), squeeze=False)
    for r, nm in enumerate(pick):
        img = np.load(os.path.join(root, "train", nm + ".npz"))["image"]
        out = pre(img)
        ax[r, 0].imshow(img, cmap="gray", vmin=0, vmax=1); ax[r, 0].set_title(f"original ({nm})")
        ax[r, 1].imshow(out, cmap="gray", vmin=0, vmax=1); ax[r, 1].set_title("preprocessed: " + str(cfg.get("preprocess")))
        ax[r, 2].imshow(out - img, cmap="seismic", vmin=-0.3, vmax=0.3); ax[r, 2].set_title("difference")
        ax[r, 3].hist(img[img > 0.02].ravel(), 128, alpha=0.6, label="original")
        ax[r, 3].hist(out[out > 0.02].ravel(), 128, alpha=0.6, label="preprocessed")
        ax[r, 3].set_title("histogram (non-background)"); ax[r, 3].legend()
        for k in range(3):
            ax[r, k].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=110)


if __name__ == "__main__":
    main()