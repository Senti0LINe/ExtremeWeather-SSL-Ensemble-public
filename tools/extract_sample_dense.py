"""Extract and Cache Dense Perception Fields and Ground Truth for Representative Case.

Tools:
    tools/extract_sample_dense.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import maximum_filter
import torch
import zarr

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ewci import data
import tools.eval_localization as el
import tools.dense_inference_v2 as di

LAT0, LON0 = 40.00448, 103.99570
H, W, S, STRIDE = 2001, 2101, 224, 56
FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="test/435/SCW2021-0072")
    ap.add_argument("--role", default="test")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--tau-prob", type=float, default=0.20)
    ap.add_argument("--tau-score", type=float, default=0.10)
    ap.add_argument("--out", default="experiments/final/dense_sample_test_435.npz")
    args = ap.parse_args()

    t_start = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[EXTRACT] Target UID: {args.uid} on device: {device}")

    # 1. Load sample record
    smp_file = FULL / f"{args.role}_samples.jsonl"
    target_sample = None
    with open(smp_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            s = json.loads(line)
            if s.get("sample_uid") == args.uid:
                target_sample = s
                break

    if not target_sample:
        raise ValueError(f"Sample UID {args.uid} not found in {smp_file}")

    obs_idx = int(target_sample["obs_index"])
    label_idx = target_sample.get("label_index")
    case_id = target_sample.get("case_id", "")
    label_utc = target_sample.get("label_utc", "")
    label_bjt = target_sample.get("label_bjt", "")
    gt_objs_raw = target_sample.get("objects", [])
    print(f"[EXTRACT] Found sample: obs={obs_idx}, label={label_idx}, GT objects={len(gt_objs_raw)}")

    # 2. Load 5-seed CINet_v2 ensemble
    models = []
    for s in range(5):
        ckpt_path = Path(f"experiments/p2_seed{s}/best.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
        m = di.CINet_v2(in_ch=41).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        models.append(m)
    print(f"[EXTRACT] Loaded {len(models)} CINet_v2 ensemble models.")

    # 3. Read observation and build 41-channel tensor
    root = zarr.open(di.STORE_PATH, mode="r")
    raw = np.asarray(root[args.role]["obs"][obs_idx - 4 : obs_idx, di.IR_SLICE, :, :])
    full_tensor, bt = di.build_full_domain_tensor(raw)

    idx_b13 = di.IR7.index("B13")
    idx_b10 = di.IR7.index("B10")
    idx_b11 = di.IR7.index("B11")

    b13_t10 = bt[3, idx_b13]
    b13_t40 = bt[0, idx_b13]
    b10_t10 = bt[3, idx_b10]
    b11_t10 = bt[3, idx_b11]

    # Ground truth radar mask
    radar_y = np.zeros((H, W), dtype=np.int8)
    if label_idx is not None and "label" in root[args.role]:
        radar_y = np.asarray(root[args.role]["label"]["y"][label_idx]).astype(np.int8)

    # 4. Dense sliding grid inference with 4-flip TTA
    rows = sorted(set(list(range(0, H - S + 1, STRIDE)) + [H - S]))
    cols = sorted(set(list(range(0, W - S + 1, STRIDE)) + [W - S]))
    patches_coords = [(r, c) for r in rows for c in cols]
    n_patches = len(patches_coords)
    w = np.sin(np.pi * (np.arange(S) + 0.5) / S) ** 2
    W2d = np.outer(w, w).astype(np.float32)

    all_p, all_hm = [], []
    with torch.no_grad():
        for i in range(0, n_patches, args.batch_size):
            chunk = patches_coords[i : i + args.batch_size]
            B = len(chunk)
            bx = np.stack([full_tensor[:, r : r + S, c : c + S] for r, c in chunk])
            bx_t = torch.from_numpy(bx).to(device)

            ens_p = torch.zeros(B, device=device)
            ens_hm = torch.zeros((B, 1, 28, 28), device=device)
            for m in models:
                c_l, h_l = m(bx_t, return_heatmap=True)
                p_acc = torch.sigmoid(c_l)
                h_acc = torch.sigmoid(h_l)
                for dims in ([3], [2], [2, 3]):
                    c_flip, h_flip = m(torch.flip(bx_t, dims=dims), return_heatmap=True)
                    p_acc = p_acc + torch.sigmoid(c_flip)
                    h_acc = h_acc + torch.flip(torch.sigmoid(h_flip), dims=dims)
                ens_p = ens_p + p_acc / 4.0
                ens_hm = ens_hm + h_acc / 4.0

            all_p.append((ens_p / len(models)).cpu().numpy())
            ens_hm = (ens_hm / len(models))
            ens_hm_224 = torch.nn.functional.interpolate(
                ens_hm, size=(S, S), mode="bilinear", align_corners=False
            ).squeeze(1).cpu().numpy()
            all_hm.append(ens_hm_224)

    all_p = np.concatenate(all_p)
    all_hm = np.concatenate(all_hm)

    # 5. 2D Hann window blending
    P_acc = np.zeros((H, W), dtype=np.float32)
    H_acc = np.zeros((H, W), dtype=np.float32)
    W_acc = np.zeros((H, W), dtype=np.float32)

    for (r, c), p_val, hm_patch in zip(patches_coords, all_p, all_hm):
        P_acc[r : r + S, c : c + S] += W2d * p_val
        H_acc[r : r + S, c : c + S] += W2d * hm_patch
        W_acc[r : r + S, c : c + S] += W2d

    P_full = P_acc / np.maximum(W_acc, 1e-6)
    H_full = H_acc / np.maximum(W_acc, 1e-6)

    # 6. Candidate peak extraction
    H_max = float(H_full.max()) if H_full.size else 1.0
    H_norm = H_full / max(H_max, 1e-6)
    score_map = P_full * H_norm

    mf = maximum_filter(score_map, size=27, mode="constant", cval=0.0)
    peaks_mask = (score_map == mf) & (score_map >= args.tau_score) & (P_full >= args.tau_prob)
    ys, xs = np.nonzero(peaks_mask)
    vals = score_map[ys, xs]
    order = np.argsort(vals)[::-1]

    # Greedy Haversine NMS with 30 km radius
    selected_peaks = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])

        # 32x32 local window
        r0, r1 = max(0, y - 16), min(H, y + 16)
        c0, c1 = max(0, x - 16), min(W, x + 16)

        w_b13_10 = b13_t10[r0:r1, c0:c1]
        w_b13_40 = b13_t40[r0:r1, c0:c1]
        w_b10_10 = b10_t10[r0:r1, c0:c1]
        w_b11_10 = b11_t10[r0:r1, c0:c1]

        min_b13_10 = float(np.nanmin(w_b13_10)) if np.isfinite(w_b13_10).any() else np.nan
        diff_b13 = w_b13_10 - w_b13_40
        delta_bt13_local = float(np.nanmin(diff_b13)) if np.isfinite(diff_b13).any() else 0.0

        diff_10_13 = w_b10_10 - w_b13_10
        b10_b13_local = float(np.nanmax(diff_10_13)) if np.isfinite(diff_10_13).any() else -99.0

        diff_11_13 = w_b11_10 - w_b13_10
        b11_b13_local = float(np.nanmin(diff_11_13)) if np.isfinite(diff_11_13).any() else 0.0

        if np.isfinite(w_b13_10).any():
            ymin, xmin = np.unravel_index(np.nanargmin(w_b13_10), w_b13_10.shape)
            y_phys = float(r0 + ymin)
            x_phys = float(c0 + xmin)
        else:
            y_phys, x_phys = float(y), float(x)

        y_dual = 0.5 * float(y) + 0.5 * y_phys
        x_dual = 0.5 * float(x) + 0.5 * x_phys

        lat = float(LAT0 - 0.01 * y_dual)
        lon = float(LON0 + 0.01 * x_dual)

        conflict = False
        for p in selected_peaks:
            if haversine_km(lat, lon, p["lat"], p["lon"]) < 30.0:
                conflict = True
                break

        if not conflict:
            p_val = float(P_full[y, x])
            score_val = float(score_map[y, x])
            hm_val = float(H_full[y, x])

            # Decaying anvil check
            is_gated = (delta_bt13_local > -0.5) and (b10_b13_local < 0.0)

            # Operational Tier determination
            if is_gated:
                tier = "GATED"
            elif p_val >= 0.60 and delta_bt13_local <= -2.0:
                tier = "RED"
            elif p_val >= 0.70:
                tier = "ORANGE"
            else:
                tier = "YELLOW"

            selected_peaks.append(
                {
                    "uid": args.uid,
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "y_nn": int(y),
                    "x_nn": int(x),
                    "y_dual": round(float(y_dual), 1),
                    "x_dual": round(float(x_dual), 1),
                    "prob": round(p_val, 4),
                    "score": round(score_val, 4),
                    "hm_val": round(hm_val, 4),
                    "bt13_local": round(min_b13_10, 1),
                    "delta_bt13_local": round(delta_bt13_local, 2),
                    "b10_b13_local": round(b10_b13_local, 2),
                    "b11_b13_local": round(b11_b13_local, 2),
                    "tier": tier,
                }
            )

    # 7. Ground truth objects formatting & Bipartite Matching
    gt_objs_clean = []
    for o in gt_objs_raw:
        if o.get("category") == "CI" or o.get("ci_label") == 1 or o.get("label") == 1:
            lat, lon = el.bbox_to_latlon(o["bbox_xyxy"])
            x1, y1, x2, y2 = o["bbox_xyxy"]
            gt_objs_clean.append(
                {
                    "object_uid": o["object_uid"],
                    "event_id": o.get("event_id", ""),
                    "bbox_xyxy": o["bbox_xyxy"],
                    "contour_xy": o.get("contour_xy", []),
                    "area_px": max(1, (x2 - x1) * (y2 - y1)),
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "source_attributes": o.get("source_attributes", []),
                }
            )

    # Run bipartite matching for kept candidates
    kept_cands = [p for p in selected_peaks if p["tier"] != "GATED"]
    h, m, fa, dists, matched_pairs = el.spatial_bipartite_matching(
        kept_cands, gt_objs_clean, tolerance_km=30.0
    )

    # Annotate hit status in candidates
    matched_coords = set((d["pred_lat"], d["pred_lon"]) for d in matched_pairs)
    for p in kept_cands:
        p["is_hit"] = (p["lat"], p["lon"]) in matched_coords
    for p in selected_peaks:
        if p["tier"] == "GATED":
            p["is_hit"] = False

    print(f"[EXTRACT] Selected peaks: {len(selected_peaks)} (Kept: {len(kept_cands)}, Gated: {len(selected_peaks)-len(kept_cands)})")
    print(f"[EXTRACT] Matching 30km: Hits={h}/{len(gt_objs_clean)}, FA={fa}, POD={h/max(1, len(gt_objs_clean)):.2%}")

    # 8. Save compressed NPZ
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        uid=args.uid,
        case_id=case_id,
        label_utc=label_utc,
        label_bjt=label_bjt,
        P_full=P_full.astype(np.float16),
        H_full=H_full.astype(np.float16),
        bt13_t10=b13_t10.astype(np.float16),
        bt13_t40=b13_t40.astype(np.float16),
        radar_y=radar_y.astype(np.int8),
        candidates_json=json.dumps(selected_peaks),
        gt_objects_json=json.dumps(gt_objs_clean),
    )
    print(f"[EXTRACT] Successfully cached dense sample to {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB) in {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
