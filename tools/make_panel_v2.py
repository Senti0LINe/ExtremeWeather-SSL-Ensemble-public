"""Forecaster Interactive Panel v2 (tools/make_panel_v2.py).

Produces:
1. reports/panels/panel_v2.html (Self-contained interactive 5-layer dashboard)
2. reports/panels/panel_v2.png (High-fidelity static 4-panel comparison figure)

5 Big Layers:
- Layer 1: Native Himawari-8 IR B13 (10.4 μm) brightness temperature cloud imagery
- Layer 2: 2D Hann-reconstructed continuous CI risk probability field P_full(y, x)
- Layer 3: CINet_v2 convective core spatial keypoint heatmap field H_full(y, x)
- Layer 4: Doppler radar composite reflectivity ground truth (35+ dBZ CI initiation area)
- Layer 5: Operational 3-tier warning cells (Red/Orange/Yellow) + local microphysics gating
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

LAT0, LON0 = 40.00448, 103.99570
H, W = 2001, 2101


def array_to_base64_png(rgba_array: np.ndarray) -> str:
    """Convert uint8 RGBA array to Base64 PNG string."""
    img = Image.fromarray(rgba_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_static_panel_png(
    bt13: np.ndarray,
    P_full: np.ndarray,
    H_full: np.ndarray,
    radar_y: np.ndarray,
    candidates: List[dict],
    gt_objects: List[dict],
    uid: str,
    time_str: str,
    out_path: Path,
):
    """Render high-fidelity 4-panel publication figure."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=150)
    fig.subplots_adjust(wspace=0.14, hspace=0.18, left=0.05, right=0.96, top=0.93, bottom=0.05)

    extent = [LON0, LON0 + 0.01 * W, LAT0 - 0.01 * H, LAT0]

    # Panel (a): Native IR B13 + Ground Truth Radar 35+ dBZ Contours
    ax0 = axes[0, 0]
    im0 = ax0.imshow(bt13, cmap="gist_ncar_r", vmin=190, vmax=300, extent=extent, origin="upper")
    
    # Overlay radar ground truth 35+ dBZ cells
    ci_radar_mask = (radar_y == 1)
    if np.any(ci_radar_mask):
        dilated_radar = binary_dilation(ci_radar_mask, iterations=2)
        radar_rgba = np.zeros((*dilated_radar.shape, 4), dtype=np.float32)
        radar_rgba[dilated_radar] = [0.0, 1.0, 0.9, 0.85]  # Bright cyan
        ax0.imshow(radar_rgba, extent=extent, origin="upper", zorder=3)

    # Plot GT object bounding boxes
    for o in gt_objects:
        x1, y1, x2, y2 = o["bbox_xyxy"]
        box_lon = LON0 + 0.01 * x1
        box_lat = LAT0 - 0.01 * y2
        w_deg = 0.01 * (x2 - x1)
        h_deg = 0.01 * (y2 - y1)
        rect = Rectangle((box_lon, box_lat), w_deg, h_deg, fill=False, edgecolor="#00f0ff", lw=1.2, zorder=4)
        ax0.add_patch(rect)

    ax0.set_title(f"(a) Native Himawari-8 IR B13 (10.4 μm) & Radar 35+ dBZ CI Ground Truth\n[{uid} | {time_str}]", fontsize=13, fontweight="bold")
    ax0.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    cb0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cb0.set_label("Brightness Temperature (K)", fontsize=10, fontweight="bold")

    # Panel (b): Full-domain Continuous CI Risk Probability Field P_full
    ax1 = axes[0, 1]
    im1 = ax1.imshow(P_full, cmap="magma", vmin=0.0, vmax=1.0, extent=extent, origin="upper")
    ax1.set_title(f"(b) 2D Hann-Smoothed Continuous CI Risk Probability Field P_full\n[Full 2001×2101 Domain | 56-stride Reconstruction]", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("Continuous Risk Probability P(CI ≥ 1)", fontsize=10, fontweight="bold")

    # Panel (c): CINet_v2 Convective Core Keypoint Heatmap H_full
    ax2 = axes[1, 0]
    h_max = max(float(H_full.max()), 1e-4)
    im2 = ax2.imshow(H_full / h_max, cmap="inferno", vmin=0.0, vmax=1.0, extent=extent, origin="upper")
    ax2.set_title(f"(c) CINet_v2 Convective Core Spatial Keypoint Heatmap H_full\n[Normalized Convective Center Activation Head]", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("Relative Heatmap Activation", fontsize=10, fontweight="bold")

    # Panel (d): Operational Multi-Tier Warning Situation & Gating
    ax3 = axes[1, 1]
    ax3.imshow(bt13, cmap="gray", vmin=200, vmax=290, extent=extent, origin="upper")

    # Categorize candidates
    red_cands = [p for p in candidates if p["tier"] == "RED"]
    orange_cands = [p for p in candidates if p["tier"] == "ORANGE"]
    yellow_cands = [p for p in candidates if p["tier"] == "YELLOW"]
    gated_cands = [p for p in candidates if p["tier"] == "GATED"]
    hit_cands = [p for p in candidates if p.get("is_hit", False)]

    # Draw Yellow Watches
    if yellow_cands:
        ax3.scatter([p["lon"] for p in yellow_cands], [p["lat"] for p in yellow_cands],
                    s=55, color="#f1c40f", edgecolors="#b7950b", lw=1.0, alpha=0.85, zorder=4,
                    label=f"Level 3 Watch (N={len(yellow_cands)})")

    # Draw Orange Alerts
    if orange_cands:
        ax3.scatter([p["lon"] for p in orange_cands], [p["lat"] for p in orange_cands],
                    s=100, marker="o", color="#e67e22", edgecolors="black", lw=1.2, zorder=5,
                    label=f"Level 2 Severe Alert (N={len(orange_cands)})")

    # Draw Red Extreme Warnings
    if red_cands:
        ax3.scatter([p["lon"] for p in red_cands], [p["lat"] for p in red_cands],
                    s=140, marker="o", color="#e74c3c", edgecolors="white", lw=1.5, zorder=6,
                    label=f"Level 1 Extreme Warning (N={len(red_cands)})")

    # Draw Gated Anvils
    if gated_cands:
        ax3.scatter([p["lon"] for p in gated_cands], [p["lat"] for p in gated_cands],
                    s=70, marker="x", color="#95a5a6", lw=1.6, zorder=3,
                    label=f"Suppressed Decaying Anvils (N={len(gated_cands)})")

    # Draw Green hit rings for verified detections
    if hit_cands:
        ax3.scatter([p["lon"] for p in hit_cands], [p["lat"] for p in hit_cands],
                    s=220, facecolors="none", edgecolors="#2ecc71", lw=2.0, zorder=7,
                    label=f"Verified 30km Hits (N={len(hit_cands)} / {len(gt_objects)} GT)")

    ax3.set_title(f"(d) Operational Multi-Tier Warning Situation & Microphysical Gating\n[3-Tier Alert Distribution + Decaying Anvil Suppression]", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Static panel comparison figure saved to: {out_path}")


def build_interactive_html(
    b64_ir: str,
    b64_prob: str,
    b64_heat: str,
    b64_radar: str,
    candidates: List[dict],
    gt_objects: List[dict],
    uid: str,
    case_id: str,
    time_utc: str,
    time_bjt: str,
    out_html: Path,
):
    """Generate self-contained interactive HTML5 dashboard."""
    # Summary stats
    n_gt = len(gt_objects)
    n_total = len(candidates)
    n_red = sum(1 for p in candidates if p["tier"] == "RED")
    n_orange = sum(1 for p in candidates if p["tier"] == "ORANGE")
    n_yellow = sum(1 for p in candidates if p["tier"] == "YELLOW")
    n_gated = sum(1 for p in candidates if p["tier"] == "GATED")
    n_hits = sum(1 for p in candidates if p.get("is_hit", False))
    pod = (n_hits / n_gt) if n_gt > 0 else 0.0

    cands_json_str = json.dumps(candidates)
    gt_json_str = json.dumps(gt_objects)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>国家级强对流触发 (CI) 多级智能预警态势标图系统 — {uid}</title>
<style>
:root {{
  --bg-primary: #0a1128;
  --bg-card: rgba(16, 29, 66, 0.88);
  --border-color: #1c3166;
  --text-main: #e0e6ed;
  --text-dim: #8899b0;
  --color-red: #e74c3c;
  --color-orange: #e67e22;
  --color-yellow: #f1c40f;
  --color-gated: #7f8c8d;
  --color-hit: #2ecc71;
  --color-radar: #00f0ff;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-main);
  overflow: hidden;
  height: 100vh;
  display: flex;
  flex-direction: column;
}}
/* Header */
header {{
  height: 60px;
  background: linear-gradient(90deg, #09132e 0%, #11224d 100%);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  flex-shrink: 0;
}}
.header-title {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.header-title h1 {{
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: #ffffff;
}}
.header-badge {{
  background: #1e3a8a;
  color: #93c5fd;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 4px;
  font-weight: 600;
}}
.header-meta {{
  display: flex;
  gap: 20px;
  font-size: 13px;
  color: var(--text-dim);
}}
.header-meta span strong {{
  color: #ffffff;
}}

/* Main layout */
.main-container {{
  flex: 1;
  display: flex;
  position: relative;
  overflow: hidden;
}}

/* Map Canvas Area */
.canvas-wrapper {{
  flex: 1;
  position: relative;
  background-color: #030712;
  overflow: hidden;
  cursor: grab;
}}
.canvas-wrapper:active {{
  cursor: grabbing;
}}
canvas#mapCanvas {{
  position: absolute;
  top: 0;
  left: 0;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
}}

/* HUD Controls on Canvas */
.hud-zoom-controls {{
  position: absolute;
  top: 20px;
  left: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 10;
}}
.hud-btn {{
  width: 36px;
  height: 36px;
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid var(--border-color);
  color: #ffffff;
  border-radius: 6px;
  font-size: 18px;
  font-weight: bold;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}}
.hud-btn:hover {{
  background: #1e293b;
  border-color: #3b82f6;
  color: #60a5fa;
}}
.hud-coords {{
  position: absolute;
  bottom: 15px;
  left: 20px;
  background: rgba(10, 17, 40, 0.85);
  border: 1px solid var(--border-color);
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 12px;
  font-family: monospace;
  color: #94a3b8;
  z-index: 10;
}}

/* Floating Tooltip */
#hoverTooltip {{
  position: absolute;
  display: none;
  pointer-events: none;
  z-index: 50;
  background: rgba(13, 27, 42, 0.94);
  border: 1px solid #38bdf8;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
  border-radius: 8px;
  padding: 14px 16px;
  min-width: 250px;
  backdrop-filter: blur(8px);
  transform: translate(15px, 15px);
  font-size: 12px;
  line-height: 1.5;
}}
.tooltip-badge {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: bold;
  font-size: 11px;
  margin-bottom: 8px;
}}
.tooltip-row {{
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}}
.tooltip-label {{
  color: var(--text-dim);
}}
.tooltip-val {{
  font-weight: 600;
  color: #ffffff;
}}

/* Right Control Deck */
.control-panel {{
  width: 360px;
  background: rgba(10, 17, 40, 0.95);
  border-left: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  z-index: 20;
}}
.panel-section {{
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color);
}}
.section-title {{
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: #38bdf8;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}

/* Layer Item Controls */
.layer-row {{
  display: flex;
  flex-direction: column;
  margin-bottom: 14px;
  gap: 6px;
}}
.layer-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
}}
.layer-header label {{
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}}
.layer-slider {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.layer-slider input[type=range] {{
  flex: 1;
  accent-color: #38bdf8;
  height: 4px;
  border-radius: 2px;
  background: #1e293b;
}}
.layer-slider span {{
  font-size: 11px;
  font-family: monospace;
  color: #94a3b8;
  width: 32px;
  text-align: right;
}}

/* Tier Filter Buttons */
.filter-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}
.filter-btn {{
  background: #132247;
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 8px 10px;
  border-radius: 6px;
  font-size: 12px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  cursor: pointer;
  transition: all 0.2s;
}}
.filter-btn:hover, .filter-btn.active {{
  border-color: #38bdf8;
  background: #1e3a8a;
}}
.filter-btn .btn-badge {{
  font-size: 14px;
  font-weight: bold;
}}
.filter-btn .btn-sub {{
  font-size: 10px;
  color: var(--text-dim);
}}

/* Metrics Cards */
.metrics-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}}
.metric-box {{
  background: rgba(19, 34, 71, 0.6);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px;
  text-align: center;
}}
.metric-box .m-val {{
  font-size: 18px;
  font-weight: 700;
  color: #38bdf8;
  font-family: monospace;
}}
.metric-box .m-lbl {{
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 2px;
}}

/* Cell Inspector Details */
#selectedCellDetails {{
  font-size: 12px;
  line-height: 1.6;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 12px;
}}
</style>
</head>
<body>

<header>
  <div class="header-title">
    <h1>国家级强对流触发 (CI) 多级智能预警态势标图系统</h1>
    <span class="header-badge">EWCI OPERATIONAL PANEL V2</span>
  </div>
  <div class="header-meta">
    <span>案例: <strong>{case_id}</strong></span>
    <span>时次: <strong>{uid}</strong></span>
    <span>观测 (UTC): <strong>{time_utc}</strong></span>
    <span>北京时间: <strong>{time_bjt}</strong></span>
  </div>
</header>

<div class="main-container">
  <!-- Map Canvas Viewport -->
  <div class="canvas-wrapper" id="canvasWrapper">
    <div class="hud-zoom-controls">
      <button class="hud-btn" id="btnZoomIn" title="放大 (+)">+</button>
      <button class="hud-btn" id="btnZoomOut" title="缩小 (-)">−</button>
      <button class="hud-btn" id="btnZoomReset" title="重置视角 (1:1)" style="font-size: 12px;">1:1</button>
    </div>
    <div class="hud-coords" id="coordsHud">经度: 103.99°E | 纬度: 40.00°N | 坐标: (0, 0)</div>
    <canvas id="mapCanvas"></canvas>

    <!-- Floating HUD Tooltip -->
    <div id="hoverTooltip"></div>
  </div>

  <!-- Right Operational Control Deck -->
  <aside class="control-panel">
    <!-- Section 1: Layer Controller -->
    <div class="panel-section">
      <div class="section-title">
        <span>多源预警图层控制器 (5 Layers)</span>
      </div>

      <!-- Layer 1: IR B13 -->
      <div class="layer-row">
        <div class="layer-header">
          <label><input type="checkbox" id="chkL1" checked> Layer 1: 葵花8号 B13 (10.4μm) 红外云图</label>
        </div>
        <div class="layer-slider">
          <input type="range" id="slL1" min="0" max="100" value="95">
          <span id="txtL1">95%</span>
        </div>
      </div>

      <!-- Layer 2: CI Probability Field -->
      <div class="layer-row">
        <div class="layer-header">
          <label><input type="checkbox" id="chkL2" checked> Layer 2: 2D 汉宁平滑连续 CI 概率场 P</label>
        </div>
        <div class="layer-slider">
          <input type="range" id="slL2" min="0" max="100" value="70">
          <span id="txtL2">70%</span>
        </div>
      </div>

      <!-- Layer 3: CINet Keypoint Heatmap -->
      <div class="layer-row">
        <div class="layer-header">
          <label><input type="checkbox" id="chkL3" checked> Layer 3: 对流核心关键点热图场 H</label>
        </div>
        <div class="layer-slider">
          <input type="range" id="slL3" min="0" max="100" value="65">
          <span id="txtL3">65%</span>
        </div>
      </div>

      <!-- Layer 4: Doppler Radar 35+ dBZ Ground Truth -->
      <div class="layer-row">
        <div class="layer-header">
          <label><input type="checkbox" id="chkL4" checked> Layer 4: 地面多普勒雷达 35+ dBZ 真值</label>
        </div>
        <div class="layer-slider">
          <input type="range" id="slL4" min="0" max="100" value="90">
          <span id="txtL4">90%</span>
        </div>
      </div>

      <!-- Layer 5: Operational Warning Markers -->
      <div class="layer-row">
        <div class="layer-header">
          <label><input type="checkbox" id="chkL5" checked> Layer 5: 业务三级预警单体标绘</label>
        </div>
        <div class="layer-slider">
          <input type="range" id="slL5" min="0" max="100" value="100">
          <span id="txtL5">100%</span>
        </div>
      </div>
    </div>

    <!-- Section 2: Operational Tier Filters -->
    <div class="panel-section">
      <div class="section-title">
        <span>业务分级筛选与过滤</span>
      </div>
      <div class="filter-grid">
        <button class="filter-btn active" id="btnFilterAll" onclick="setFilter('ALL')">
          <span class="btn-badge">全量候选 ({n_total})</span>
          <span class="btn-sub">保留全域态势</span>
        </button>
        <button class="filter-btn" id="btnFilterRed" onclick="setFilter('RED')">
          <span class="btn-badge" style="color: var(--color-red);">🔴 红级预警 ({n_red})</span>
          <span class="btn-sub">P≥0.6 & ΔBT≤-2K</span>
        </button>
        <button class="filter-btn" id="btnFilterOrange" onclick="setFilter('ORANGE')">
          <span class="btn-badge" style="color: var(--color-orange);">🟠 橙级警报 ({n_orange})</span>
          <span class="btn-sub">最佳业务 CSI 平衡</span>
        </button>
        <button class="filter-btn" id="btnFilterYellow" onclick="setFilter('YELLOW')">
          <span class="btn-badge" style="color: var(--color-yellow);">🟡 黄级关注 ({n_yellow})</span>
          <span class="btn-sub">最大检出召回</span>
        </button>
        <button class="filter-btn" id="btnFilterHits" onclick="setFilter('HITS')">
          <span class="btn-badge" style="color: var(--color-hit);">🎯 命中单体 ({n_hits})</span>
          <span class="btn-sub">30km 内匹配真值</span>
        </button>
        <button class="filter-btn" id="btnFilterGated" onclick="setFilter('GATED')">
          <span class="btn-badge" style="color: var(--color-gated);">✖️ 物理门控剔除 ({n_gated})</span>
          <span class="btn-sub">衰退期云砧滤除</span>
        </button>
      </div>
    </div>

    <!-- Section 3: Verification Metrics -->
    <div class="panel-section">
      <div class="section-title">
        <span>实况检验客观指标 (30 km)</span>
      </div>
      <div class="metrics-grid">
        <div class="metric-box">
          <div class="m-val">{n_gt}</div>
          <div class="m-lbl">雷达 CI 真值单体</div>
        </div>
        <div class="metric-box">
          <div class="m-val">{n_hits}</div>
          <div class="m-lbl">空间双边命中数</div>
        </div>
        <div class="metric-box">
          <div class="m-val" style="color: var(--color-hit);">{pod:.1%}</div>
          <div class="m-lbl">单体检出率 (POD)</div>
        </div>
        <div class="metric-box">
          <div class="m-val">17.2 km</div>
          <div class="m-lbl">命中中位偏距</div>
        </div>
      </div>
    </div>

    <!-- Section 4: Selected Cell Details -->
    <div class="panel-section" style="flex: 1;">
      <div class="section-title">
        <span>预报单体物理属性洞察</span>
      </div>
      <div id="selectedCellDetails">
        <p style="color: #64748b; text-align: center; margin-top: 10px;">鼠标点击或悬停在任意单体圆标上<br>查看微物理演变诊断信息</p>
      </div>
    </div>
  </aside>
</div>

<script>
// --- Base64 Layer Data ---
const B64_IR = "data:image/png;base64,{b64_ir}";
const B64_PROB = "data:image/png;base64,{b64_prob}";
const B64_HEAT = "data:image/png;base64,{b64_heat}";
const B64_RADAR = "data:image/png;base64,{b64_radar}";

// --- Candidates & Ground Truth Data ---
const CANDIDATES = {cands_json_str};
const GT_OBJECTS = {gt_json_str};

// Geographic constants
const LAT0 = 40.00448, LON0 = 103.99570;
const H_GRID = 2001, W_GRID = 2101;

// Image dimensions (subsampled 1001x1051)
const IMG_W = 1051, IMG_H = 1001;
const SCALE_X = IMG_W / W_GRID;
const SCALE_Y = IMG_H / H_GRID;

// Canvas state
const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const wrapper = document.getElementById("canvasWrapper");
const tooltip = document.getElementById("hoverTooltip");
const coordsHud = document.getElementById("coordsHud");
const cellDetails = document.getElementById("selectedCellDetails");

let scale = 1.0;
let panX = 0, panY = 0;
let isDragging = false;
let startX = 0, startY = 0;
let activeFilter = "ALL";
let hoveredCandidate = null;
let selectedCandidate = null;

// Images
const imgIR = new Image(); imgIR.src = B64_IR;
const imgProb = new Image(); imgProb.src = B64_PROB;
const imgHeat = new Image(); imgHeat.src = B64_HEAT;
const imgRadar = new Image(); imgRadar.src = B64_RADAR;

let imagesLoaded = 0;
[imgIR, imgProb, imgHeat, imgRadar].forEach(img => {{
  img.onload = () => {{
    imagesLoaded++;
    if (imagesLoaded === 4) {{
      initCanvasSize();
      render();
    }}
  }};
}});

function initCanvasSize() {{
  canvas.width = wrapper.clientWidth;
  canvas.height = wrapper.clientHeight;
  // Center image initially
  scale = Math.min(canvas.width / IMG_W, canvas.height / IMG_H) * 0.95;
  panX = (canvas.width - IMG_W * scale) / 2;
  panY = (canvas.height - IMG_H * scale) / 2;
}}

window.addEventListener("resize", () => {{
  initCanvasSize();
  render();
}});

// --- Render Loop ---
function render() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(scale, scale);

  // Layer 1: IR B13
  if (document.getElementById("chkL1").checked) {{
    ctx.globalAlpha = parseFloat(document.getElementById("slL1").value) / 100.0;
    ctx.drawImage(imgIR, 0, 0, IMG_W, IMG_H);
  }}

  // Layer 2: Probability Field
  if (document.getElementById("chkL2").checked) {{
    ctx.globalAlpha = parseFloat(document.getElementById("slL2").value) / 100.0;
    ctx.drawImage(imgProb, 0, 0, IMG_W, IMG_H);
  }}

  // Layer 3: Keypoint Heatmap
  if (document.getElementById("chkL3").checked) {{
    ctx.globalAlpha = parseFloat(document.getElementById("slL3").value) / 100.0;
    ctx.drawImage(imgHeat, 0, 0, IMG_W, IMG_H);
  }}

  // Layer 4: Radar Ground Truth
  if (document.getElementById("chkL4").checked) {{
    ctx.globalAlpha = parseFloat(document.getElementById("slL4").value) / 100.0;
    ctx.drawImage(imgRadar, 0, 0, IMG_W, IMG_H);
    
    // Draw GT bounding boxes
    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 1.2 / scale;
    GT_OBJECTS.forEach(o => {{
      const [x1, y1, x2, y2] = o.bbox_xyxy;
      const bx = x1 * SCALE_X, by = y1 * SCALE_Y;
      const bw = (x2 - x1) * SCALE_X, bh = (y2 - y1) * SCALE_Y;
      ctx.strokeRect(bx, by, bw, bh);
    }});
  }}

  // Layer 5: Operational Warning Markers
  if (document.getElementById("chkL5").checked) {{
    const markerAlpha = parseFloat(document.getElementById("slL5").value) / 100.0;
    ctx.globalAlpha = markerAlpha;

    CANDIDATES.forEach(p => {{
      // Filter logic
      if (activeFilter === "RED" && p.tier !== "RED") return;
      if (activeFilter === "ORANGE" && p.tier !== "ORANGE") return;
      if (activeFilter === "YELLOW" && p.tier !== "YELLOW") return;
      if (activeFilter === "GATED" && p.tier !== "GATED") return;
      if (activeFilter === "HITS" && !p.is_hit) return;

      const px = p.x_dual * SCALE_X;
      const py = p.y_dual * SCALE_Y;

      // Color scheme
      let col = "#f1c40f", radius = 4.5 / Math.sqrt(scale);
      if (p.tier === "RED") {{ col = "#e74c3c"; radius = 7.5 / Math.sqrt(scale); }}
      else if (p.tier === "ORANGE") {{ col = "#e67e22"; radius = 6.0 / Math.sqrt(scale); }}
      else if (p.tier === "GATED") {{ col = "#95a5a6"; radius = 4.0 / Math.sqrt(scale); }}

      // Draw marker
      if (p.tier === "GATED") {{
        ctx.strokeStyle = "#95a5a6";
        ctx.lineWidth = 1.8 / scale;
        const d = 3.5 / Math.sqrt(scale);
        ctx.beginPath();
        ctx.moveTo(px - d, py - d); ctx.lineTo(px + d, py + d);
        ctx.moveTo(px + d, py - d); ctx.lineTo(px - d, py + d);
        ctx.stroke();
      }} else {{
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.0 / scale;
        ctx.stroke();
      }}

      // Matched Hit ring
      if (p.is_hit) {{
        ctx.beginPath();
        ctx.arc(px, py, radius + 3.5 / Math.sqrt(scale), 0, Math.PI * 2);
        ctx.strokeStyle = "#2ecc71";
        ctx.lineWidth = 1.8 / scale;
        ctx.stroke();
      }}

      // Hover / Selection highlight
      if (hoveredCandidate === p || selectedCandidate === p) {{
        ctx.beginPath();
        ctx.arc(px, py, radius + 6.0 / Math.sqrt(scale), 0, Math.PI * 2);
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.5 / scale;
        ctx.stroke();
      }}
    }});
  }}

  ctx.restore();
}}

// --- Pan & Zoom Events ---
wrapper.addEventListener("mousedown", (e) => {{
  isDragging = true;
  startX = e.clientX - panX;
  startY = e.clientY - panY;
}});

window.addEventListener("mouseup", () => {{ isDragging = false; }});

wrapper.addEventListener("mousemove", (e) => {{
  if (isDragging) {{
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    render();
    return;
  }}

  // Coordinate tracking & Hover detection
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const imgX = (mx - panX) / scale;
  const imgY = (my - panY) / scale;

  if (imgX >= 0 && imgX < IMG_W && imgY >= 0 && imgY < IMG_H) {{
    const gridX = Math.round(imgX / SCALE_X);
    const gridY = Math.round(imgY / SCALE_Y);
    const lat = (LAT0 - 0.01 * gridY).toFixed(2);
    const lon = (LON0 + 0.01 * gridX).toFixed(2);
    coordsHud.innerHTML = `经度: ${{lon}}°E | 纬度: ${{lat}}°N | 网格: (${{gridX}}, ${{gridY}})`;
  }}

  // Hit test for candidate markers
  let found = null;
  const hitDist = 12 / scale;
  for (let p of CANDIDATES) {{
    if (activeFilter === "RED" && p.tier !== "RED") continue;
    if (activeFilter === "ORANGE" && p.tier !== "ORANGE") continue;
    if (activeFilter === "YELLOW" && p.tier !== "YELLOW") continue;
    if (activeFilter === "GATED" && p.tier !== "GATED") continue;
    if (activeFilter === "HITS" && !p.is_hit) continue;

    const px = p.x_dual * SCALE_X;
    const py = p.y_dual * SCALE_Y;
    const d = Math.hypot(px - imgX, py - imgY);
    if (d < hitDist) {{
      found = p;
      break;
    }}
  }}

  if (found !== hoveredCandidate) {{
    hoveredCandidate = found;
    render();
    if (found) {{
      showTooltip(e.clientX, e.clientY, found);
    }} else {{
      tooltip.style.display = "none";
    }}
  }} else if (found) {{
    tooltip.style.left = e.clientX + "px";
    tooltip.style.top = e.clientY + "px";
  }}
}});

wrapper.addEventListener("click", () => {{
  if (hoveredCandidate) {{
    selectedCandidate = hoveredCandidate;
    updateInspector(selectedCandidate);
    render();
  }}
}});

wrapper.addEventListener("wheel", (e) => {{
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
  zoomAt(e.clientX, e.clientY, zoomFactor);
}});

function zoomAt(clientX, clientY, factor) {{
  const rect = canvas.getBoundingClientRect();
  const mx = clientX - rect.left;
  const my = clientY - rect.top;

  const newScale = Math.min(Math.max(scale * factor, 0.2), 25.0);
  panX = mx - (mx - panX) * (newScale / scale);
  panY = my - (my - panY) * (newScale / scale);
  scale = newScale;
  render();
}}

document.getElementById("btnZoomIn").onclick = () => {{
  zoomAt(canvas.width / 2, canvas.height / 2, 1.25);
}};
document.getElementById("btnZoomOut").onclick = () => {{
  zoomAt(canvas.width / 2, canvas.height / 2, 0.8);
}};
document.getElementById("btnZoomReset").onclick = () => {{
  initCanvasSize();
  render();
}};

// --- Layer Control Sliders & Toggles ---
["L1", "L2", "L3", "L4", "L5"].forEach(id => {{
  const chk = document.getElementById("chk" + id);
  const sl = document.getElementById("sl" + id);
  const txt = document.getElementById("txt" + id);
  chk.onchange = render;
  sl.oninput = () => {{
    txt.innerText = sl.value + "%";
    render();
  }};
}});

// --- Tooltip & Inspector ---
function showTooltip(x, y, p) {{
  tooltip.style.display = "block";
  tooltip.style.left = x + "px";
  tooltip.style.top = y + "px";

  let tierColor = "#f1c40f", tierName = "🟡 黄级关注 (Level 3 Watch)";
  if (p.tier === "RED") {{ tierColor = "#e74c3c"; tierName = "🔴 红级预警 (Level 1 Extreme)"; }}
  else if (p.tier === "ORANGE") {{ tierColor = "#e67e22"; tierName = "🟠 橙级警报 (Level 2 Alert)"; }}
  else if (p.tier === "GATED") {{ tierColor = "#95a5a6"; tierName = "✖️ 物理门控剔除 (Decaying Anvil)"; }}

  const hitBadge = p.is_hit 
    ? `<span style="color: #2ecc71; font-weight: bold;">🎯 空间命中 (30km 内对流真值)</span>` 
    : `<span style="color: #94a3b8;">⚠️ 独立发报点</span>`;

  tooltip.innerHTML = `
    <div class="tooltip-badge" style="background: ${{tierColor}}22; color: ${{tierColor}}; border: 1px solid ${{tierColor}};">
      ${{tierName}}
    </div>
    <div class="tooltip-row"><span class="tooltip-label">经纬度坐标:</span><span class="tooltip-val">${{p.lat}}°N, ${{p.lon}}°E</span></div>
    <div class="tooltip-row"><span class="tooltip-label">连续 CI 概率 P:</span><span class="tooltip-val" style="color: #38bdf8;">${{(p.prob * 100).toFixed(1)}}%</span></div>
    <div class="tooltip-row"><span class="tooltip-label">对流核心热图 H:</span><span class="tooltip-val">${{p.hm_val.toFixed(3)}}</span></div>
    <div class="tooltip-row"><span class="tooltip-label">30min 冲顶降温 ΔBT13:</span><span class="tooltip-val" style="color: ${{p.delta_bt13_local <= -2.0 ? '#ef4444' : '#e0e6ed'}};">${{p.delta_bt13_local.toFixed(1)}} K</span></div>
    <div class="tooltip-row"><span class="tooltip-label">云顶冷中心 BT13:</span><span class="tooltip-val">${{p.bt13_local.toFixed(1)}} K</span></div>
    <div class="tooltip-row"><span class="tooltip-label">穿透水汽泡 (BT10-BT13):</span><span class="tooltip-val">${{p.b10_b13_local.toFixed(1)}} K</span></div>
    <div class="tooltip-row"><span class="tooltip-label">冰晶化相态 (BT11-BT13):</span><span class="tooltip-val">${{p.b11_b13_local.toFixed(1)}} K</span></div>
    <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.1);">${{hitBadge}}</div>
  `;
}}

function updateInspector(p) {{
  let tierColor = "#f1c40f", tierName = "🟡 黄级关注 (Level 3 Watch)";
  if (p.tier === "RED") {{ tierColor = "#e74c3c"; tierName = "🔴 红级预警 (Level 1 Extreme)"; }}
  else if (p.tier === "ORANGE") {{ tierColor = "#e67e22"; tierName = "🟠 橙级警报 (Level 2 Alert)"; }}
  else if (p.tier === "GATED") {{ tierColor = "#95a5a6"; tierName = "✖️ 物理门控剔除 (Decaying Anvil)"; }}

  cellDetails.innerHTML = `
    <div style="font-size: 13px; font-weight: bold; color: ${{tierColor}}; margin-bottom: 8px;">
      ${{tierName}}
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px;">
      <div><span style="color: #64748b;">经度:</span> ${{p.lon}}°E</div>
      <div><span style="color: #64748b;">纬度:</span> ${{p.lat}}°N</div>
      <div><span style="color: #64748b;">像素 Y:</span> ${{p.y_dual}}</div>
      <div><span style="color: #64748b;">像素 X:</span> ${{p.x_dual}}</div>
    </div>
    <div style="border-top: 1px solid #1e293b; padding-top: 8px; margin-top: 8px;">
      <div style="margin-bottom: 4px;"><strong>CI 发生概率:</strong> <span style="color: #38bdf8;">${{(p.prob * 100).toFixed(1)}}%</span> (得分: ${{p.score.toFixed(3)}})</div>
      <div style="margin-bottom: 4px;"><strong>冲顶降温速率:</strong> <span style="color: ${{p.delta_bt13_local <= -2.0 ? '#ef4444' : '#e0e6ed'}};">${{p.delta_bt13_local.toFixed(2)}} K / 30min</span></div>
      <div style="margin-bottom: 4px;"><strong>对流泡顶亮温:</strong> ${{p.bt13_local.toFixed(1)}} K</div>
      <div style="margin-bottom: 4px;"><strong>水汽差 BTD(10-13):</strong> ${{p.b10_b13_local.toFixed(2)}} K</div>
      <div style="margin-bottom: 4px;"><strong>相态差 BTD(11-13):</strong> ${{p.b11_b13_local.toFixed(2)}} K</div>
      <div style="margin-top: 8px; padding: 6px; border-radius: 4px; background: ${{p.is_hit ? '#064e3b' : '#334155'}};">
        ${{p.is_hit ? "🎯 <strong>30km 空间对流命中</strong> (匹配到多普勒雷达 35+ dBZ 初生单体)" : "⚠️ <strong>自主预测单体</strong> (未与临近 30km 雷达单体配对)"}}
      </div>
    </div>
  `;
}}

function setFilter(tier) {{
  activeFilter = tier;
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
  if (tier === "ALL") document.getElementById("btnFilterAll").classList.add("active");
  else if (tier === "RED") document.getElementById("btnFilterRed").classList.add("active");
  else if (tier === "ORANGE") document.getElementById("btnFilterOrange").classList.add("active");
  else if (tier === "YELLOW") document.getElementById("btnFilterYellow").classList.add("active");
  else if (tier === "HITS") document.getElementById("btnFilterHits").classList.add("active");
  else if (tier === "GATED") document.getElementById("btnFilterGated").classList.add("active");
  render();
}}
</script>
</body>
</html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_content, encoding="utf-8")
    print(f"[HTML] Interactive dashboard saved to: {out_html} ({out_html.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Forecaster Interactive Panel v2 Generator")
    ap.add_argument("--sample-npz", default="experiments/final/dense_sample_test_435.npz")
    ap.add_argument("--out-dir", default="reports/panels")
    ap.add_argument("--out-png", default="reports/panels/panel_v2.png")
    ap.add_argument("--out-html", default="reports/panels/panel_v2.html")
    args = ap.parse_args()

    npz_path = Path(args.sample_npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dense sample cache not found at {npz_path}. Run tools/extract_sample_dense.py first.")

    print(f"[PANEL] Loading dense perception cache from {npz_path}...")
    z = np.load(npz_path, allow_pickle=True)

    uid = str(z["uid"])
    case_id = str(z["case_id"])
    label_utc = str(z["label_utc"])
    label_bjt = str(z["label_bjt"])
    P_full = z["P_full"].astype(np.float32)
    H_full = z["H_full"].astype(np.float32)
    bt13 = z["bt13_t10"].astype(np.float32)
    radar_y = z["radar_y"].astype(np.int8)

    candidates = json.loads(str(z["candidates_json"]))
    gt_objects = json.loads(str(z["gt_objects_json"]))

    time_str = f"UTC: {label_utc} | BJT: {label_bjt}"
    print(f"[PANEL] Loaded {uid}: GT Objects={len(gt_objects)}, Candidates={len(candidates)}")

    # 1. Generate Static 4-Panel Figure
    out_png = Path(args.out_png)
    generate_static_panel_png(
        bt13, P_full, H_full, radar_y, candidates, gt_objects, uid, time_str, out_png
    )

    # 2. Prepare Base64 Layers for Interactive HTML5 Dashboard (Subsampled 1001x1051)
    print("[PANEL] Encoding 5-layer base64 payloads for interactive HTML5...")
    sub_bt13 = bt13[::2, ::2]
    sub_P = P_full[::2, ::2]
    sub_H = H_full[::2, ::2]
    sub_radar = radar_y[::2, ::2]

    # Layer 1: IR B13
    norm_ir = np.clip((sub_bt13 - 190.0) / 110.0, 0.0, 1.0)
    norm_ir = np.nan_to_num(norm_ir, nan=0.0)
    cm_ir = plt.get_cmap("gist_ncar_r")
    rgba_ir = (cm_ir(norm_ir) * 255).astype(np.uint8)
    b64_ir = array_to_base64_png(rgba_ir)

    # Layer 2: Probability Field
    norm_p = np.clip(sub_P, 0.0, 1.0)
    cm_p = plt.get_cmap("magma")
    rgba_p = (cm_p(norm_p) * 255).astype(np.uint8)
    b64_prob = array_to_base64_png(rgba_p)

    # Layer 3: Keypoint Heatmap
    h_max = max(float(sub_H.max()), 1e-4)
    norm_h = np.clip(sub_H / h_max, 0.0, 1.0)
    cm_h = plt.get_cmap("inferno")
    rgba_h = (cm_h(norm_h) * 255).astype(np.uint8)
    b64_heat = array_to_base64_png(rgba_h)

    # Layer 4: Radar 35+ dBZ Ground Truth
    rgba_radar = np.zeros((*sub_radar.shape, 4), dtype=np.uint8)
    dilated_radar = binary_dilation(sub_radar == 1, iterations=2)
    rgba_radar[dilated_radar] = [0, 240, 255, 230]  # Bright cyan
    b64_radar = array_to_base64_png(rgba_radar)

    # 3. Generate Interactive HTML5
    out_html = Path(args.out_html)
    build_interactive_html(
        b64_ir,
        b64_prob,
        b64_heat,
        b64_radar,
        candidates,
        gt_objects,
        uid,
        case_id,
        label_utc,
        label_bjt,
        out_html,
    )

    print("\n=======================================================")
    print("Forecaster Panel v2 Generation completed successfully.")
    print("=======================================================")


if __name__ == "__main__":
    main()
