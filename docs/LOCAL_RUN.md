# Running this repository locally

This release is a **byte-for-byte replica** of the working repository (only three
internal operational documents and one email address were removed). The code,
the scripts, the reported numbers and the line-number citations in
[EXPERIMENTS.md](EXPERIMENTS.md) are unchanged.

Because of that, the original absolute data paths are still in the sources.
`tools/configure_paths.py` repoints them in one command — see below.

---

## 1. What works with no data at all

Clone, install four libraries, and you can already use the flagship deliverable:

```bash
pip install numpy scipy matplotlib pillow
python3 tools/selfcheck.py          # installation self-check
python3 tools/make_panel_v2.py      # rebuild the forecaster panel
open reports/panels/panel_v2.html   # self-contained, no CDN, no server
```

`tools/make_panel_v2.py` needs **only** numpy / scipy / matplotlib / pillow — no
torch, no GPU, no dataset. It renders from `experiments/final/dense_sample_test_435.npz`
(10 MB, shipped in this repo).

Also available immediately: every published number in `experiments/final/*.json`,
and the delivered panels and figures under `reports/`.

## 2. What ships in this repo (so you can skip re-computing it)

| artifact | size | lets you skip |
|---|---|---|
| 10 trained checkpoints (v1 ×5 seeds, v2 ×5 seeds) | 57 MB | all training |
| v1/v2 TTA predictions for dev / benchmark / test (25 npz) | 5.5 MB | all inference |
| physical feature matrices (train / dev / val / test_eval) | 18 MB | feature building |
| fitted LightGBM models (`gbdt_cache.pkl`) | 14 MB | GBDT training |
| dense sample for one test timestamp | 10 MB | dense inference for the panel |
| dual-localisation cache | 0.5 MB | re-running the keypoint forward pass |
| v2 ensemble archives (`cinet_v2_{dev,bench,test}_predictions.npz`) | 1.2 MB | window-level ensemble evaluation and the dev-only selection of alpha / tau_obj |
| dense candidate peaks (`dense_candidates_{dev,benchmark,test}.json`) | 38 MB | **all GPU dense inference** for the object-level chain |

With these plus the annotations you can reproduce the published tables without
training anything. The object-level chain is CPU-only once the candidate caches
are present:

~~~
python tools/local_physics_gating.py        # ~2 min, no GPU
python tools/eval_operational_tiers.py      # ~1 min, selects tau on dev then freezes
python tools/diagnose_far.py                # ~3 min, FAR root-cause diagnosis (E16)
~~~

`tools/diagnose_far.py` is read-only: it trains nothing and changes no operating
point. It validates its own matcher against `eval_localization.spatial_bipartite_matching`
and its AP against `eval_operational_tiers.compute_object_ap` before reporting, so
its numbers are directly comparable to the published ones.

## 3. What you must supply yourself

| requirement | detail |
|---|---|
| **Himawari-8 crop cube** | `dataset/dataset.zarr`, **239 GB**. Not redistributable here — GitHub caps a single file at 100 MB and a repository at ~1 GB. |
| **CIDS annotations** | `experiments/ci_det_v2/annotations_stage1_bg/` (24 MB) and `annotations_full/` (744 MB). Course-provided labels; not included. |
| **normalisation climatology** | `dataset/norm_stats.npz` (1.3 KB). Ships with the dataset. |
| **Python** | 3.10+ (developed and measured on 3.11.16) |
| **PyTorch** | 2.1+ (measured on 2.14.1+cu130) |
| **GPU** | optional for the panel and the self-check; **strongly recommended** for training and inference. Measured on RTX 4090 48 GB. CPU works but is roughly 30× slower for the dense stage. |
| **RAM** | 32 GB recommended for the dense full-domain stage (2001 × 2101 float fields × several copies) |
| **disk** | ~1 GB for the repo plus ~250 GB if you host the dataset locally |

Install the rest with:

```bash
pip install -r requirements.txt
```

### Pointing the code at your data

```bash
python3 tools/configure_paths.py --dry-run --data-root /my/DATASAT --work-root /my/work
python3 tools/configure_paths.py           --data-root /my/DATASAT --work-root /my/work
```

Expected layout under `--data-root`:

```
<data-root>/dataset/dataset.zarr
<data-root>/dataset/norm_stats.npz
<data-root>/experiments/ci_det_v2/annotations_stage1_bg/{train,val}_{patches,samples}.jsonl
<data-root>/experiments/ci_det_v2/annotations_full/{val,test}_{patches,samples}.jsonl
```

`--work-root` is only needed for the ERA5 side experiments (`data/era5/`,
`.cdsapirc`, `env_proxy.sh`).

## 4. Full reproduction without Slurm

```bash
bash tools/local_run.sh          # self-check + panel, then the full chain if data is present
bash tools/local_run.sh full     # force the full chain
```

`tools/local_run.sh` is an **additive** entry point: it runs exactly the commands
in `slurm/seeds_tta.slurm`, `slurm/test5.slurm`, `slurm/train_v2_parallel.slurm`
and `tools/predict_all_seeds_v2.sh`, but **serially on one device**. The original
scripts are untouched.

Measured cost on one RTX 4090 (from the shipped training logs):

* v2 training: **37.5 min per seed**, 6 epochs → ~3.1 h for five seeds serially
* v1 training: ~19 min per seed → ~1.6 h for five seeds serially
* feature building: ~3 min per split with 8 workers
* dense inference and the object-level evaluation scale with GPU count; the
  published run used four GPUs in parallel

Overrides: `PYTHON`, `WORKERS`, `GPUS`, `DEVICE`, `MINUTES_V1`, `MINUTES_V2`.

## 5. Known traps (read before running)

1. **`tools/extract_sample_dense.py` defaults to `--device cuda:3`.** On a single-GPU
   or CPU machine pass `--device cuda:0` or `--device cpu`. `local_run.sh` does this
   for you.
2. **`tools/dense_inference_v2.py` and `tools/eval_localization_dual.py` default to
   `--gpus 1,2,3,4`.** Pass `--gpus 0` on a single-GPU box.
3. **`tools/predict_all_seeds_v2.sh` (`GPUS=(1 2 4 5 6)`) and
   `tools/run_multi_gpu_parallel.sh` (`GPUS=(1 2 3 4 5)`) hardcode GPU arrays** that
   cannot be overridden by a flag. Use `tools/local_run.sh` instead, or edit the
   array. These two scripts are left exactly as in the original.
4. **Training has a wall-clock budget** (`--minutes`, 45 for v1 and 60 for v2 in the
   original jobs). On slower hardware it silently stops mid-epoch. All five
   published v2 seeds finished 6/6 epochs in ~37.5 min, so the published numbers
   are not truncated — but your run may be. `local_run.sh` defaults to 600 min.
5. **The original reproduction commands in the README use `sbatch`.** There is no
   Slurm on a laptop; use `tools/local_run.sh`.
6. **`ewci/tensors.py` reads `norm_stats.npz` at import time.** Without the dataset,
   importing it fails — this is why `tools/selfcheck.py` deliberately does not
   import it.
7. **Only three scripts have an explicit CPU fallback** (`train_cnn.py`,
   `train_cnn_v2.py`, `predict_cnn_v2.py`). The dense stage does fall back to CPU
   internally, but expect it to be very slow.

## 6. What cannot work here

* **Training or inference without the dataset.** 239 GB of satellite imagery is
  not in this repository and cannot be.
* **Re-evaluation without the annotations.** The window-level and object-level
  evaluators read labels from `annotations_full/*_samples.jsonl` and
  `*_patches.jsonl`. The shipped prediction archives let you skip inference, but
  the labels still have to come from the annotations.
* **Real-time operational use.** No live satellite feed, no live radar mosaic.
  See the Limitations section of the README.

## 7. Verifying a run

`experiments/final/*.json` are the authoritative published numbers. After a
re-run, compare against them; the headline values are listed in the README and
every one of them is traceable to a specific JSON field.

Note that `docs/EXPERIMENTS.md` **E14** documents 12 confirmed defects in the
current pipeline (heatmap supervision, truncated objects, weak gating, NMS
radius, truncated PR-AUC, and others). They are recorded, **not fixed** in this
release. Reproducing the published numbers therefore reproduces those defects
too — that is intentional, so the numbers stay comparable.