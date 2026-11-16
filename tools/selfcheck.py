"""Installation self-check — runs WITHOUT the satellite dataset.

Verifies dependencies, shipped artifacts, model forward passes, checkpoint
loading and the metric implementations. Exits non-zero on any failure.

    python3 tools/selfcheck.py

Note: ewci.tensors reads the dataset climatology at import time, so it is
deliberately not imported here. Tensor construction from raw Zarr patches is
only exercisable once the dataset is available (see docs/LOCAL_RUN.md).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

RESULTS: list[tuple[bool, str, str]] = []


def record(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" — " + detail) if detail else ""))


def check_dependencies() -> None:
    print("\n[1/5] Dependencies")
    required = {
        "numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn",
        "torch": "torch", "matplotlib": "matplotlib", "PIL": "pillow",
        "joblib": "joblib", "pandas": "pandas",
    }
    optional = {"lightgbm": "lightgbm", "zarr": "zarr", "xarray": "xarray"}
    for mod, pkg in required.items():
        try:
            m = importlib.import_module(mod)
            record(True, pkg, getattr(m, "__version__", "installed"))
        except Exception as exc:
            record(False, pkg, str(exc)[:70])
    for mod, pkg in optional.items():
        try:
            m = importlib.import_module(mod)
            record(True, pkg + " (needed for data loading / GBDT)",
                   getattr(m, "__version__", "installed"))
        except Exception:
            print("  [SKIP] %s not installed — only needed with the dataset" % pkg)


def check_artifacts() -> None:
    print("\n[2/5] Shipped artifacts")
    weights = [PROJECT / "experiments/p1_cnn/best.pt"]
    weights += [PROJECT / ("experiments/p1_seed%d/best.pt" % s) for s in (1, 2, 3, 4)]
    weights += [PROJECT / ("experiments/p2_seed%d/best.pt" % s) for s in range(5)]
    missing = [str(w.relative_to(PROJECT)) for w in weights if not w.exists()]
    record(not missing, "10 trained checkpoints (v1 x5, v2 x5)",
           "all present" if not missing else "missing: " + ", ".join(missing))

    sample = PROJECT / "experiments/final/dense_sample_test_435.npz"
    record(sample.exists(), "dense sample for panel regeneration",
           "%.1f MB" % (sample.stat().st_size / 1e6) if sample.exists() else "absent")

    metrics = sorted((PROJECT / "experiments/final").glob("*metrics*.json"))
    record(len(metrics) >= 5, "published metric JSONs", "%d files" % len(metrics))

    for rel in ("splits_v1.json", "splits_test_eval.json"):
        record((PROJECT / rel).exists(), rel)


def check_model() -> None:
    print("\n[3/5] Model forward pass (random input, no data needed)")
    try:
        import torch
        from ewci.model_v2 import CINet_v2
    except Exception as exc:
        record(False, "import CINet_v2", str(exc)[:90])
        return

    torch.manual_seed(0)
    model = CINet_v2(in_ch=41, width=32, heatmap_size=28)
    x = torch.randn(2, 41, 224, 224)
    try:
        cls, hm = model(x, return_heatmap=True)
        ok = tuple(cls.shape) == (2,) and tuple(hm.shape) == (2, 1, 28, 28)
        record(ok, "CINet_v2 forward", "cls%s heatmap%s" % (tuple(cls.shape), tuple(hm.shape)))
        n_params = sum(p.numel() for p in model.parameters())
        record(True, "parameter count", "%.2f M" % (n_params / 1e6))
    except Exception as exc:
        record(False, "CINet_v2 forward", str(exc)[:90])
        return

    ck = PROJECT / "experiments/p2_seed0/best.pt"
    if ck.exists():
        try:
            sd = torch.load(ck, map_location="cpu")
            model.load_state_dict(sd)
            record(True, "load shipped v2 seed0 checkpoint")
        except Exception as exc:
            record(False, "load shipped v2 seed0 checkpoint", str(exc)[:90])

    try:
        from ewci.model import CINet
        record(True, "import CINet v1")
    except Exception as exc:
        record(False, "import CINet v1", str(exc)[:90])


def check_metrics() -> None:
    print("\n[4/5] Metric implementations (known-answer test)")
    try:
        import numpy as np
        from ewci import metrics as M
    except Exception as exc:
        record(False, "import ewci.metrics", str(exc)[:90])
        return

    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    s = M.skill(y, p, 0.5)
    expected = {"POD": 0.5, "FAR": 0.5, "CSI": 1 / 3, "ETS": 0.0, "HSS": 0.0, "BIAS": 1.0}
    for k, v in expected.items():
        record(abs(s[k] - v) < 1e-6, "%s = %.4f" % (k, v), "got %.6f" % s[k])
    record(s["hits"] == 1 and s["false_alarms"] == 1 and s["misses"] == 1
           and s["correct_negatives"] == 1, "contingency table h/f/m/cn = 1/1/1/1")

    r = M.ranking(y, p)
    record(abs(r["ROC_AUC"] - 0.75) < 1e-6, "ROC_AUC = 0.75", "got %.6f" % r["ROC_AUC"])
    _, _, _, ece = M.reliability(y, p, n_bins=10)
    record(0.0 <= ece <= 1.0, "ECE in [0,1]", "got %.6f" % ece)


def check_panel_path() -> None:
    print("\n[5/5] Zero-data deliverable path")
    try:
        import numpy as np
        z = np.load(PROJECT / "experiments/final/dense_sample_test_435.npz", allow_pickle=True)
        keys = list(z.keys())
        record(len(keys) > 0, "dense sample readable", "keys: " + ", ".join(keys[:8]))
    except Exception as exc:
        record(False, "dense sample readable", str(exc)[:90])
    for mod in ("matplotlib", "PIL", "scipy.ndimage"):
        try:
            importlib.import_module(mod)
            record(True, "panel renderer dependency %s" % mod)
        except Exception as exc:
            record(False, "panel renderer dependency %s" % mod, str(exc)[:70])


def main() -> int:
    print("=" * 68)
    print("ExtremeWeather-SSL-Ensemble — installation self-check")
    print("repo: %s" % PROJECT)
    print("=" * 68)
    check_dependencies()
    check_artifacts()
    check_model()
    check_metrics()
    check_panel_path()

    failed = [r for r in RESULTS if not r[0]]
    print("\n" + "=" * 68)
    print("%d checks, %d passed, %d failed" % (len(RESULTS), len(RESULTS) - len(failed), len(failed)))
    print("=" * 68)
    print("""
What you can do right now, with no dataset:
  * open reports/panels/panel_v2.html in a browser (self-contained)
  * rebuild that panel:  python3 tools/make_panel_v2.py
  * read every published number in experiments/final/*.json

What needs the Himawari-8 dataset + CIDS annotations (see docs/LOCAL_RUN.md):
  * feature building, training, inference, and all re-evaluation
""")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())