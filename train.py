"""Train the CSSNet baseline on Synapse.

    python train.py --config config.yaml --out runs/exp1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import build_model, load_config, set_seed
from dataset import SliceDataset
from losses import CSSNetLoss
from metrics import dice_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N slices (smoke test)")
    ap.add_argument("--max_epochs", type=int, default=None, help="override config (smoke test)")
    ap.add_argument("--resume", action="store_true", help="continue from last.pth if the session dropped")
    ap.add_argument("--data_root", default=None, help="override config data_root (e.g. Colab local copy of Drive data)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    if args.data_root:
        cfg["data_root"] = args.data_root
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    json.dump({**cfg, "device": str(device)}, open(os.path.join(args.out, "config_used.json"), "w"), indent=2)

    train_ds = SliceDataset(cfg["data_root"], cfg["list_dir"], "train", cfg["img_size"], args.limit)
    val_ds = SliceDataset(cfg["data_root"], cfg["list_dir"], "val", cfg["img_size"], args.limit)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = build_model(cfg["num_classes"]).to(device)
    criterion = CSSNetLoss(cfg["num_classes"], cfg["topk_threshold"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["base_lr"], weight_decay=cfg["weight_decay"])
    total_iters = cfg["max_epochs"] * len(train_dl)
    patience = cfg["early_stop_patience"]
    min_delta = cfg["early_stop_min_delta"]

    log, best_val, stale, it, start_epoch = [], -1.0, 0, 0, 0
    last_path = os.path.join(args.out, "last.pth")
    log_path = os.path.join(args.out, "log.json")
    if args.resume and os.path.isfile(last_path):
        ckpt = _load(last_path, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, it, best_val, stale = ckpt["epoch"] + 1, ckpt["it"], ckpt["best_val"], ckpt["stale"]
        if os.path.isfile(log_path):
            log = json.load(open(log_path))
        print(f"resume from epoch {start_epoch}, best val dice {best_val:.4f}")

    for epoch in range(start_epoch, cfg["max_epochs"]):
        model.train()
        t0, losses = time.time(), []
        for batch in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            x, y = batch["image"].to(device), batch["label"].to(device)
            if cfg["lr_schedule"] == "poly":
                for g in optimizer.param_groups:
                    g["lr"] = cfg["base_lr"] * (1 - it / total_iters) ** 0.9
            logits = model(x)[0]  # main output only (paper sec. 3.4)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            it += 1

        val_dice = validate(model, val_dl, cfg["num_classes"], device)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_dice": val_dice, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        if val_dice > best_val + min_delta:
            best_val = val_dice
            stale = 0
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
            rec["best"] = True
        else:
            stale += 1
            rec["best"] = False
            rec["stale_epochs"] = stale
        log.append(rec)
        print(json.dumps(rec))
        json.dump(log, open(log_path, "w"), indent=1)
        torch.save({"epoch": epoch, "it": it, "best_val": best_val, "stale": stale, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, last_path)
        if stale >= patience:
            print(f"early stop at epoch {epoch}: val mean Dice unchanged for {patience} epochs (best {best_val:.4f})")
            break
    print(f"done. best val mean dice {best_val:.4f}")


@torch.no_grad()
def validate(model, dl, num_classes, device):
    """Mean over val slices of the per-slice mean organ Dice (2D, at network resolution)."""
    model.eval()
    scores = []
    for batch in dl:
        pred = torch.argmax(model(batch["image"].to(device))[0], dim=1).cpu().numpy()
        for p, g in zip(pred, batch["label"].numpy()):
            d, _ = dice_iou(p, g, num_classes)
            if not np.all(np.isnan(d)):
                scores.append(np.nanmean(d))
    return float(np.mean(scores)) if scores else 0.0


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    main()
