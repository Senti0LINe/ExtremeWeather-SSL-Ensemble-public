"""Data access for the CI benchmark.

Design rules
------------
* Time-based split only: train = 2021-03-01..2021-04-30, val = 2021-05.
* The official benchmark is the 30 validation timestamps selected with
  seed=2021 (9,180 windows). Windows are never sampled randomly.
* A window is positive iff it contains at least one CI object.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr

STORE_PATH = "/data1/group3/DATASAT/dataset/dataset.zarr"
ANNOTATIONS_DIR = Path(
    "/data1/group3/DATASAT/experiments/ci_det_v2/annotations_stage1_bg"
)

STORE_BANDS = ("B03", "B07", "B08", "B09", "B10", "B11", "B13", "B14", "B16")
IR7 = ("B08", "B09", "B10", "B11", "B13", "B14", "B16")
IR_INDEX = {b: STORE_BANDS.index(b) for b in IR7}
IR_SLICE = [IR_INDEX[b] for b in IR7]

BTD_PAIRS = (
    ("B08", "B10"),
    ("B08", "B11"),
    ("B10", "B13"),
    ("B11", "B13"),
    ("B13", "B14"),
    ("B14", "B16"),
)

PATCH_SIZE = 224
FILL_VALUE = -32768
SCALE = 0.01  # Kelvin per raw count


def load_samples(role: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    path = ANNOTATIONS_DIR / (role + "_samples.jsonl")
    with open(path, "r") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["sample_uid"]] = rec
    return out


def load_patches(role: str) -> List[dict]:
    out: List[dict] = []
    path = ANNOTATIONS_DIR / (role + "_patches.jsonl")
    with open(path, "r") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def select_validation_uids(
    samples: Dict[str, dict], maximum: int = 30, seed: int = 2021
) -> List[str]:
    """Official round-robin timestamp selector across meteorological cases."""
    groups: Dict[str, List[str]] = {}
    for uid, sample in samples.items():
        groups.setdefault(sample["case_id"], []).append(uid)
    rng = random.Random(seed)
    cases = sorted(groups)
    for case in cases:
        groups[case].sort()
        rng.shuffle(groups[case])
    rng.shuffle(cases)
    chosen = set()
    while len(chosen) < maximum:
        for case in cases:
            if groups[case]:
                chosen.add(groups[case].pop())
                if len(chosen) == maximum:
                    break
    return [uid for uid in samples if uid in chosen]


def benchmark_patches(val_max_timestamps: int = 30, seed: int = 2021) -> List[dict]:
    samples = load_samples("val")
    chosen = set(select_validation_uids(samples, val_max_timestamps, seed))
    return [p for p in load_patches("val") if p["sample_uid"] in chosen]


def has_ci(patch: dict) -> int:
    return int(any(o.get("label", 0) == 1 for o in patch.get("objects", [])))


def load_patches_from(root, role: str) -> List[dict]:
    out: List[dict] = []
    path = Path(root) / (role + "_patches.jsonl")
    with open(path, "r") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


class ObsReader:
    """Lazy per-process reader for one zarr role (open once per worker)."""

    def __init__(self, role: str, store_path: str = STORE_PATH):
        self.role = role
        self.store_path = store_path
        self._obs = None

    @property
    def obs(self):
        if self._obs is None:
            root = zarr.open(self.store_path, mode="r")
            self._obs = root[self.role]["obs"]
        return self._obs

    def read_patch(self, patch: dict, bands: List[int] = IR_SLICE) -> np.ndarray:
        oi = int(patch["obs_index"])
        row = int(patch["row"])
        col = int(patch["col"])
        size = int(patch.get("patch_size", PATCH_SIZE))
        return np.asarray(
            self.obs[oi - 4 : oi, bands, row : row + size, col : col + size]
        )


def split_by_timestamp(
    patches: List[dict], fraction: float = 0.2, seed: int = 7
) -> Tuple[List[dict], List[dict]]:
    uids = sorted({p["sample_uid"] for p in patches})
    rng = random.Random(seed)
    rng.shuffle(uids)
    n_hold = max(1, int(round(len(uids) * fraction)))
    hold = set(uids[:n_hold])
    a = [p for p in patches if p["sample_uid"] in hold]
    b = [p for p in patches if p["sample_uid"] not in hold]
    return a, b
