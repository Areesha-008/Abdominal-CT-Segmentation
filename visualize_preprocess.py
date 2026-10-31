"""Before/after figures + histograms for the report (sections 5.1 and 6.1).

Runs every config in --configs on the same slices and writes one PNG per config:
  preprocess_examples_exp2.png (CLAHE), preprocess_examples_exp3.png (unsharp)

Press Run in VS Code, or:
  python visualize_preprocess.py --configs config_exp2.yaml config_exp3.yaml --data_root "C:/path/to/cv_dataset"
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


def make_figure(cfg_path, data_root, n, out):
    cfg = load_config(cfg_path)
    root = data_root or cfg["data_root"]
    pre = build_preprocess(cfg)
    names = read_list(cfg["list_dir"], "train.txt")
    pick = names[:: max(1, len(names) // n)][:n]  # same slices for every config
    fig, ax = plt.subplots(n, 4, figsize=(14, 3.6 * n), squeeze=False)
    for r, nm in enumerate(pick):
        img = np.load(os.path.join(root, "train", nm + ".npz"))["image"]
        res = pre(img)
        v = max(0.05, float(np.percentile(np.abs(res - img), 99.5)))
        ax[r, 0].imshow(img, cmap="gray", vmin=0, vmax=1); ax[r, 0].set_title(f"original ({nm})", fontsize=9)
        ax[r, 1].imshow(res, cmap="gray", vmin=0, vmax=1); ax[r, 1].set_title("preprocessed: " + str(cfg.get("preprocess")), fontsize=8)
        ax[r, 2].imshow(res - img, cmap="seismic", vmin=-v, vmax=v); ax[r, 2].set_title(f"difference (+/-{v:.2f})", fontsize=9)
        ax[r, 3].hist(img[img > 0.02].ravel(), 128, alpha=0.6, label="original")
        ax[r, 3].hist(res[res > 0.02].ravel(), 128, alpha=0.6, label="preprocessed")
        ax[r, 3].set_title("histogram (non-background)", fontsize=9); ax[r, 3].legend()
        for k in range(3):
            ax[r, k].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["config_exp2.yaml", "config_exp3.yaml"])
    ap.add_argument("--data_root", default="C:/Users/User/Downloads/cv_dataset")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out_dir", default=".")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for c in a.configs:
        tag = os.path.splitext(os.path.basename(c))[0].replace("config_", "")
        make_figure(c, a.data_root, a.n, os.path.join(a.out_dir, f"preprocess_examples_{tag}.png"))


if __name__ == "__main__":
    main()