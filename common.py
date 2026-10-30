import os
import random
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSSNet"))
from networks.Seg import CONFIGS, MLPmedical  # noqa: E402

from dataset import resize_image, resize_label  # noqa: E402


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(num_classes):
    # Same construction as the released train.py, with the class count set correctly (it defaulted to 6).
    return MLPmedical(CONFIGS["R50-ViT-B_16"], num_class=num_classes)


@torch.no_grad()
def predict_volume(model, volume, img_size, device):
    """volume: (D,H,W) float32. Mirrors the released test_single_volume: resize each slice to
    img_size, predict, argmax, resize the prediction back to native resolution."""
    model.eval()
    pred = np.zeros(volume.shape, dtype=np.int64)
    for i, s in enumerate(volume):
        x = torch.from_numpy(resize_image(s, img_size))[None, None].to(device)
        out = torch.argmax(model(x)[0], dim=1)[0].cpu().numpy()
        pred[i] = resize_label(out, s.shape[0])
    return pred
