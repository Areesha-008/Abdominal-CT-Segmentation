"""Loss from the paper (sec. 3.4): L = L_TopK + L_Dice, applied to the main output."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKLoss(nn.Module):
    """Cross-entropy averaged only over pixels whose true-class probability is below `threshold` (eq. 15)."""

    def __init__(self, threshold=0.8):
        super().__init__()
        self.threshold = threshold

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=1)
        log_p_true = log_p.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,H,W)
        hard = log_p_true.exp() < self.threshold
        if hard.sum() == 0:
            return logits.sum() * 0.0
        return -(log_p_true[hard]).mean()


class SoftDiceLoss(nn.Module):
    """1 - mean over classes of soft Dice (eq. 16)."""

    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        p = torch.softmax(logits, dim=1)
        t = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (p * t).sum(dim=(0, 2, 3))
        denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class CSSNetLoss(nn.Module):
    def __init__(self, num_classes, topk_threshold=0.8):
        super().__init__()
        self.topk = TopKLoss(topk_threshold)
        self.dice = SoftDiceLoss(num_classes)

    def forward(self, logits, target):
        return self.topk(logits, target) + self.dice(logits, target)
