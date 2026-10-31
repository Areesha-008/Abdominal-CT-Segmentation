"""Synapse slices (train/val) and volumes (test).

Train/val: one 512x512 slice per .npz (image float32 in [0,1], label 0..8).
Test: one .npy.h5 per case holding the full volume (D, 512, 512).
"""
import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

ORGANS = ["aorta", "gallbladder", "kidney_left", "kidney_right", "liver", "pancreas", "spleen", "stomach"]


def read_list(list_dir, name):
    with open(os.path.join(list_dir, name)) as f:
        return [l.strip() for l in f if l.strip()]


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """Same __call__ as CSSNet/datasets/dataset_synapse.py (rot/flip or rotate, then zoom)."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.4:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.4:
            image, label = random_rotate(image, label)
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        return {"image": image, "label": label.long()}


class SliceDataset(Dataset):
    """Train or val slices. Train uses the released augmentation; val only resizes."""

    def __init__(self, data_root, list_dir, split, img_size, limit=None, preprocess=None):
        assert split in ("train", "val")
        self.preprocess = preprocess
        self.names = read_list(list_dir, f"{split}.txt")[:limit]
        self.dir = os.path.join(data_root, "train")
        self.img_size = img_size
        self.augment = RandomGenerator(output_size=[img_size, img_size]) if split == "train" else None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        d = np.load(os.path.join(self.dir, self.names[idx] + ".npz"))
        image, label = d["image"], d["label"]
        if self.preprocess is not None:
            image = self.preprocess(image)  # before augmentation/resize, same as for test volumes
        if self.augment is not None:
            return self.augment({"image": image, "label": label})
        return {"image": resize_image(image, self.img_size)[None], "label": resize_label(label, self.img_size).astype(np.int64)}


class VolumeDataset(Dataset):
    """Test volumes at native 512x512."""

    def __init__(self, data_root, list_dir, limit=None, preprocess=None):
        self.preprocess = preprocess
        self.names = read_list(list_dir, "test_vol.txt")[:limit]
        self.dir = os.path.join(data_root, "test")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with h5py.File(os.path.join(self.dir, self.names[idx] + ".npy.h5"), "r") as f:
            image, label = f["image"][:], f["label"][:]
        if self.preprocess is not None:
            image = np.stack([self.preprocess(sl) for sl in image])
        return {"image": image, "label": label.astype(np.int64), "case": self.names[idx]}


def resize_image(image, size):
    h, w = image.shape
    return image if (h, w) == (size, size) else zoom(image, (size / h, size / w), order=3).astype(np.float32)


def resize_label(label, size):
    h, w = label.shape
    return label if (h, w) == (size, size) else zoom(label, (size / h, size / w), order=0)