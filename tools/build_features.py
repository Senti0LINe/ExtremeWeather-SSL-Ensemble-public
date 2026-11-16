"""Build tabular physics features for a named split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ewci import data, features  # noqa: E402

FULL_ANN = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
_READER = None


def _init(role: str) -> None:
    global _READER
    _READER = data.ObsReader(role)


def _work(patch: dict) -> np.ndarray:
    return features.extract(_READER.read_patch(patch))


def _full_patches(role: str):
    return data.load_patches_from(FULL_ANN, role)


def select_patches(split: str):
    if split == "train":
        return "train", data.load_patches("train")
    if split == "val":  # official 30-timestamp benchmark
        return "val", data.benchmark_patches()
    if split == "dev":
        sp = json.loads((PROJECT / "splits_v1.json").read_text())
        uids = set(sp["dev"]["early_stop_uids"])
        print("dev timestamps: %d" % len(uids))
        return "val", [p for p in _full_patches("val") if p["sample_uid"] in uids]
    if split == "test":
        sp = json.loads((PROJECT / "splits_v1.json").read_text())
        uids = set(sp["test"]["uids"])
        return "test", [p for p in _full_patches("test") if p["sample_uid"] in uids]
    if split == "test_eval":
        sp = json.loads((PROJECT / "splits_test_eval.json").read_text())
        uids = set(sp["uids"])
        return "test", [p for p in _full_patches("test") if p["sample_uid"] in uids]
    raise SystemExit("unknown split " + split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "dev", "test", "test_eval"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    role, patches = select_patches(args.split)
    if args.limit:
        patches = patches[: args.limit]
    patches.sort(key=lambda p: (int(p["obs_index"]), int(p["row"]), int(p["col"])))
    t0 = time.time()
    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(args.workers, initializer=_init, initargs=(role,)) as pool:
            X = pool.map(_work, patches, chunksize=16)
    else:
        _init(role)
        X = [_work(p) for p in patches]
    X = np.asarray(X, dtype=np.float32)

    y = np.array([data.has_ci(p) for p in patches], dtype=np.int8)
    n_obj = np.array([len(p.get("objects", [])) for p in patches], dtype=np.int16)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        y=y,
        names=np.array(features.FEATURE_NAMES),
        uid=np.array([p["sample_uid"] for p in patches]),
        case=np.array([p.get("case_id", "") for p in patches]),
        utc=np.array([p.get("label_utc", "") for p in patches]),
        label_index=np.array([int(p.get("label_index", -1)) for p in patches], dtype=np.int32),
        row=np.array([int(p["row"]) for p in patches], dtype=np.int32),
        col=np.array([int(p["col"]) for p in patches], dtype=np.int32),
        n_obj=n_obj,
    )
    dt = time.time() - t0
    print("wrote %s X=%s pos=%d neg=%d %.1fs" % (out, X.shape, int(y.sum()), int((1 - y).sum()), dt))


if __name__ == "__main__":
    main()
