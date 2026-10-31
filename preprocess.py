"""Intensity preprocessing for Experiment 2 / 3 (applied identically to train, val and test).

Images are float32 in [0, 1] (CT already windowed), shape (H, W).
Select the technique in the config:

preprocess:
  name: clahe          # none | clahe | gamma | gaussian | unsharp
  clip_limit: 2.0      # clahe
  tile_grid: 8         # clahe (tile_grid x tile_grid tiles)
"""
import cv2
import numpy as np


def _clahe(img, clip_limit=2.0, tile_grid=8):
    # 8-bit (256 bins) is the standard CLAHE setting. With 16-bit input OpenCV's per-bin clip
    # threshold collapses to 1, so clip_limit stops having any effect (checked for 2..16).
    u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)).apply(u8)
    return eq.astype(np.float32) / 255.0


def _gamma(img, gamma=0.8):
    return np.power(np.clip(img, 0, 1), gamma).astype(np.float32)


def _gaussian(img, sigma=1.0):
    return cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)


def _unsharp(img, sigma=1.0, amount=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img + amount * (img - blur), 0, 1).astype(np.float32)


_FUNCS = {"clahe": _clahe, "gamma": _gamma, "gaussian": _gaussian, "unsharp": _unsharp}


def build_preprocess(cfg):
    """Return a callable (H, W) float32 -> (H, W) float32, or None for the baseline."""
    pcfg = dict(cfg.get("preprocess") or {})
    name = pcfg.pop("name", "none")
    if name in (None, "none"):
        return None
    if name not in _FUNCS:
        raise ValueError(f"unknown preprocess '{name}', choose from {sorted(_FUNCS)} or 'none'")
    fn = _FUNCS[name]
    return lambda img: fn(np.ascontiguousarray(img, dtype=np.float32), **pcfg)