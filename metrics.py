"""Per-organ Dice and IoU on a whole label volume (or slice)."""
import numpy as np


def dice_iou(pred, gt, num_classes):
    """Returns two arrays of length num_classes-1 (organs 1..C-1).

    If an organ is absent from the ground truth of this case, its entry is NaN so that
    it is excluded from case averages (use np.nanmean) instead of being scored 0 or 1.
    """
    dice = np.full(num_classes - 1, np.nan)
    iou = np.full(num_classes - 1, np.nan)
    for c in range(1, num_classes):
        p, g = pred == c, gt == c
        n_g = g.sum()
        if n_g == 0:
            continue
        inter = np.logical_and(p, g).sum()
        n_p = p.sum()
        dice[c - 1] = 2 * inter / (n_p + n_g)
        iou[c - 1] = inter / (n_p + n_g - inter)
    return dice, iou
