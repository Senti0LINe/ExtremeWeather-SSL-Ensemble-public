"""Build the CNN input tensor from a raw zarr patch.

Channels (41):
  28 = 7 IR bands x 4 frames, standardised with the dataset climatology
   6 = brightness-temperature differences at the last frame (Kelvin, /5)
   7 = per-band change between the first and last frame (Kelvin, /2)

Unlike the legacy pipeline these are computed on *physical* brightness
temperatures, so every plane keeps its meteorological meaning.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .data import BTD_PAIRS, FILL_VALUE, IR7, SCALE, STORE_BANDS

_NORM = np.load("/data1/group3/DATASAT/dataset/norm_stats.npz")
_BANDS = list(_NORM["bands"])
_MEAN = np.array([_NORM["mean"][_BANDS.index(b)] for b in IR7], dtype=np.float32)
_STD = np.array([_NORM["std"][_BANDS.index(b)] for b in IR7], dtype=np.float32)
_IDX = {b: i for i, b in enumerate(IR7)}


def build_input(raw: np.ndarray, size: int = 128) -> np.ndarray:
    """raw int16 [4, 7, H, W] -> float32 [41, size, size]."""
    raw = np.asarray(raw)
    bt = raw.astype(np.float32) * SCALE
    bt[raw == FILL_VALUE] = np.nan

    # fill missing with the per-plane valid mean (0 after standardisation)
    for t in range(bt.shape[0]):
        for c in range(bt.shape[1]):
            plane = bt[t, c]
            m = np.isfinite(plane)
            if m.any():
                plane[~m] = plane[m].mean()
            else:
                plane[:] = 0.0

    normed = (bt - _MEAN[None, :, None, None]) / _STD[None, :, None, None]
    last = bt[-1]
    first = bt[0]

    planes = [normed.reshape(-1, *normed.shape[-2:])]  # 28 channels
    planes.append(np.stack([(last[_IDX[a]] - last[_IDX[b]]) / 5.0 for a, b in BTD_PAIRS]))  # 6
    planes.append((last - first) / 2.0)  # 7

    x = np.concatenate(planes, axis=0).astype(np.float32)

    if x.shape[-1] != size:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(x)[None]
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        x = t[0].numpy()
    return x
