"""Evaluate a checkpoint on the 12 test volumes; write results.json and overlay PNGs.

    python evaluate.py --config config.yaml --run runs/exp1
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from common import build_model, load_config, predict_volume
from dataset import ORGANS, VolumeDataset
from preprocess import build_preprocess
from metrics import dice_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", required=True, help="run folder containing best.pth")
    ap.add_argument("--checkpoint", default="best.pth")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test volumes (smoke test)")
    ap.add_argument("--data_root", default=None, help="override config data_root")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["num_classes"]).to(device)
    ckpt_path = os.path.join(args.run, args.checkpoint)
    if not os.path.isfile(ckpt_path) and args.checkpoint == "best.pth":
        ckpt_path = os.path.join(args.run, "last.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"no checkpoint in {args.run} (looked for best.pth / last.pth)")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)

    ds = VolumeDataset(cfg["data_root"], cfg["list_dir"], args.limit, preprocess=build_preprocess(cfg))
    overlay_slices = {c: i for c, i in cfg["overlay_slices"]}
    os.makedirs(os.path.join(args.run, "overlays"), exist_ok=True)

    per_case, all_dice, all_iou = {}, [], []
    for sample in tqdm(ds, desc="test volumes"):
        pred = predict_volume(model, sample["image"], cfg["img_size"], device)
        d, j = dice_iou(pred, sample["label"], cfg["num_classes"])
        per_case[sample["case"]] = {"dice": d.tolist(), "iou": j.tolist()}
        all_dice.append(d)
        all_iou.append(j)
        if sample["case"] in overlay_slices:
            k = overlay_slices[sample["case"]]
            save_overlay(sample["image"][k], sample["label"][k], pred[k], os.path.join(args.run, "overlays", f"{sample['case']}_slice{k:03d}.png"))

    mean_dice, mean_iou = np.nanmean(all_dice, axis=0), np.nanmean(all_iou, axis=0)
    results = {
        "checkpoint": args.checkpoint,
        "n_cases": len(ds),
        "per_organ": {o: {"dice": float(mean_dice[i]), "iou": float(mean_iou[i])} for i, o in enumerate(ORGANS)},
        "mean_dice": float(mean_dice.mean()),
        "mean_iou": float(mean_iou.mean()),
        "per_case": per_case,
    }
    json.dump(results, open(os.path.join(args.run, "results.json"), "w"), indent=1)

    print(f"{'organ':14s} {'Dice':>8s} {'IoU':>8s}")
    for o in ORGANS:
        print(f"{o:14s} {results['per_organ'][o]['dice']:8.4f} {results['per_organ'][o]['iou']:8.4f}")
    print(f"{'MEAN':14s} {results['mean_dice']:8.4f} {results['mean_iou']:8.4f}")


def save_overlay(image, gt, pred, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, mask in zip(axes, ["image", "ground truth", "prediction"], [None, gt, pred]):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        if mask is not None:
            ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="tab10", vmin=0, vmax=9, alpha=0.5, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()