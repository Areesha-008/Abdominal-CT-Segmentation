"""Synapse slices (train/val) and volumes (test).

Train/val: one 512x512 slice per .npz (image float32 in [0,1], label 0..8).
Test: one .npy.h5 per case holding the full volume (D, 512, 512).
"""
import os
import sys

import h5py
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import Dataset

_CSSNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSSNet")
sys.path.insert(0, _CSSNET)
# Colab ships HuggingFace `datasets`, which shadows CSSNet/datasets in sys.modules
for _k in [k for k in sys.modules if k == "datasets" or k.startswith("datasets.")]:
    del sys.modules[_k]
from datasets.dataset_synapse import RandomGenerator  # noqa: E402  (released rot/flip + zoom augmentation)

ORGANS = ["aorta", "gallbladder", "kidney_left", "kidney_right", "liver", "pancreas", "spleen", "stomach"]


def read_list(list_dir, name):
    with open(os.path.join(list_dir, name)) as f:
        return [l.strip() for l in f if l.strip()]


class SliceDataset(Dataset):
    """Train or val slices. Train uses the released augmentation; val only resizes."""

    def __init__(self, data_root, list_dir, split, img_size, limit=None):
        assert split in ("train", "val")
        self.names = read_list(list_dir, f"{split}.txt")[:limit]
        self.dir = os.path.join(data_root, "train")
        self.img_size = img_size
        self.augment = RandomGenerator(output_size=[img_size, img_size]) if split == "train" else None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        d = np.load(os.path.join(self.dir, self.names[idx] + ".npz"))
        image, label = d["image"], d["label"]
        if self.augment is not None:
            return self.augment({"image": image, "label": label})
        return {"image": resize_image(image, self.img_size)[None], "label": resize_label(label, self.img_size).astype(np.int64)}


class VolumeDataset(Dataset):
    """Test volumes at native 512x512."""

    def __init__(self, data_root, list_dir, limit=None):
        self.names = read_list(list_dir, "test_vol.txt")[:limit]
        self.dir = os.path.join(data_root, "test")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with h5py.File(os.path.join(self.dir, self.names[idx] + ".npy.h5"), "r") as f:
            image, label = f["image"][:], f["label"][:]
        return {"image": image, "label": label.astype(np.int64), "case": self.names[idx]}


def resize_image(image, size):
    h, w = image.shape
    return image if (h, w) == (size, size) else zoom(image, (size / h, size / w), order=3).astype(np.float32)


def resize_label(label, size):
    h, w = label.shape
    return label if (h, w) == (size, size) else zoom(label, (size / h, size / w), order=0)
