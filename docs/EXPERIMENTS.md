# Experiment log — CI detection on Himawari-8 / CIDS

This file records every meaningful experiment, including the ones that failed
and the wrong turns we took. It is written so that a supervisor or a new
contributor can see *why* the current design looks the way it does.

All numbers below are on the official benchmark (30 validation timestamps,
9,180 windows, 1,175 positives = 12.8%) unless stated otherwise.

---

## 0. The task

Detect **convection initiation (CI)** from 4 Himawari-8 infrared frames
(T-40, T-30, T-20, T-10 minutes). Input: a 224 x 224 window (2.24 deg,
about 250 km). Output: the probability that the window contains a CI object.

Why it is hard: the object has a median area of 72 pixels (about 10 km),
i.e. 0.14% of the window; about 25% of CI objects still have warm tops; and
the same scene is full of much colder mature clouds.

---

## 1. Data and splits

Source: /data1/group3/DATASAT (Himawari crop + CIDS labels), annotations in
annotations_stage1_bg (train) and annotations_full (val, test).

| split | period (UTC) | timestamps | windows | positive rate | use |
|---|---|---|---|---|---|
| train | 2021-03-01..04-30 | 2,972 | 28,586 | 79% | fit only |
| dev | 2021-05-01..05-13 (not the benchmark) | 100 used | 30,600 | 14.2% | model choice, threshold, calibration |
| benchmark | 30 timestamps inside dev period | 30 | 9,180 | 12.8% | final comparison with v0 only |
| test | 2021-05-14..05-31 | 2,221 | 679,626 | 24.0% | opened once, at the very end |

Rules: a timestamp never appears in two splits; the threshold is chosen on dev
only; the benchmark is never used for tuning; the test set is frozen in
splits_v1.json (sha256 28ad6cf0...).

Why the positive rates differ: the training windows were built around each CI
object, the dev/test windows are a uniform 17 x 18 grid (306 per timestamp).
This prior shift is real and is handled with class weights plus a dev-tuned
threshold, not by resampling the evaluation set.

---

## 2. Audit of the legacy system (v0)

v0 reported AUC 0.6745, PR-AUC 0.2474, CSI 19.87% at tau=0.64, POD 37.6%,
FAR 70.4%, Brier 0.2535. It had four components (3D autoencoder, FixMatch,
LightGBM, stacking). Reading the code and running ablations found:

| issue | evidence |
|---|---|
| ERA5 branch is random noise | ERA5Loader() has no data_dir, so load_field() calls _generate_synthetic(); /data1/group3/DATASAT/era5 does not exist. 27 of the input channels change every batch. |
| the deep encoder depends on that noise | same checkpoint, 800 benchmark windows: AUC 0.721 with the synthetic fields, 0.523 with zeros, 0.768 with the branch removed. Reconstruction R2 +0.335 / -18.6 / -0.01. |
| temporal information is not used | feeding the last frame 4 times gives AUC 0.7379 vs 0.7213 for the real 4-frame sequence; reversing time changes nothing (0.7222). AdaptiveAvgPool3d((1,None,None)) averages the 4 frames away. |
| global average pooling hides the object | 224 -> 112 resize, then AdaptiveAvgPool2d(1) to z in R^128: the 0.14%-area object is averaged out. |
| the "self-supervised" reconstruction is degenerate | reconstructing 13 x 112 x 112 from a 128-d pooled vector reaches R2 = 0.335 (mean field). |
| FixMatch is not semi-supervised | weak/strong views come from the same labelled batch; there is no unlabelled pool; max-probability pseudo-labels are dominated by confident background. |
| stacking hurts | the meta-net is trained on in-sample base predictions; the LightGBM model is not saved in the checkpoint, so the reported number cannot be reproduced; "best model" is just the last epoch. |
| the threshold is tuned on the report set | best_tau is selected on the same 9,180 windows that are reported. |

Conclusion: the four components were real implementations, but they were wired
in a way that destroyed the signal. This is why v1 starts from a floor that is
easy to trust, and adds the deep components back one at a time.

---

## 3. Experiments, in order

### E0 — reproduce the benchmark first
Rebuilt select_validation_uids(seed=2021): 30 timestamps, 9,180 windows,
1,175 positives — exactly the v0 counts. Every later comparison is on the same
windows.

### E1 — the leakage mistake (important)
A 51-feature LightGBM evaluated with a random 5-fold split inside May gave
AUC 0.825, PR-AUC 0.457, CSI 33.6%. That looked like a 14-point win over v0.
It was wrong: the random split put windows from the same timestamp in both
train and test.

Correct time split (train Mar-Apr, test May): AUC 0.709. Within May, grouped by
timestamp: 0.796; random by window: 0.880. So ~0.08 of the apparent skill was
leakage, and ~0.09 more is period shift.

Lesson recorded in the README: never split windows randomly.

### E2 — how much does data volume help?
| training data | May AUC |
|---|---|
| March only (11,654) | 0.650 |
| April only (16,932) | 0.717 |
| March + April (28,586) | 0.709 |
| March + April, random 50% | 0.713 |

Saturated. More of the same windows does not help; the bottleneck is the
representation, not the raw count.

### E3 — is the model just memorising locations?
Location-only model (row, col): AUC 0.544. No.

### E4 — the 51 physical features
Brightness-temperature statistics, cold-area fractions, cooling rate, cold-area
growth, 6 band differences, gradient/texture, centre-crop statistics.
Frozen protocol (threshold from dev, evaluated on the benchmark):
AUC 0.7072 [0.690, 0.724], PR-AUC 0.2893, CSI 20.5% [18.9, 22.1],
POD 44.5%, FAR 72.5%.
At fixed false-positive rate: POFD 10% -> POD 33.2%; POFD 20% -> POD 48.3%.

### E5 — calibration
Raw tree probabilities are severely over-confident: ECE 0.276, Brier 0.2095
(predicted 0.6, observed 0.12). Isotonic regression fitted on dev only:
ECE 0.011, Brier 0.1035, ranking unchanged. This is the largest practical gain
so far: the number can now be read as a probability.

### E6 — the deep model
A compact CNN (41 physical planes: 7 bands x 4 frames + 6 band differences +
7 temporal differences; 128 px; GroupNorm + GeM pooling; ~1.2 M parameters)
trained on one RTX 4090 for 6 epochs (19 minutes), batch 64, BCE with
pos_weight = neg/pos, flips + brightness jitter + cutout.

| model | dev AUC (100 ts) | benchmark AUC | benchmark PR-AUC |
|---|---|---|---|
| physics features + GBDT | 0.7341 | 0.7072 | 0.2893 |
| compact CNN | 0.7695 | 0.7211 | 0.3757 |
| 0.75 CNN + 0.25 GBDT (weight from dev) | 0.7757 | 0.7329 | 0.3835 |

The CNN is better on both splits, and much better in the low-false-alarm
region (PR-AUC 0.376 vs 0.289). It is kept.

### E7 — calibration of the ensemble
Isotonic regression fitted on dev only (never on the benchmark or test):
ECE 0.246 -> 0.012, Brier 0.1671 -> 0.0945.

Operating points on the benchmark, threshold chosen on dev:
- at tau = 0.28: POD 37.5%, FAR 49.1%, CSI 27.5%
- at POFD <= 10%: POD 44.4%, CSI 26.4%
- at POFD <= 20%: POD 53.4%, CSI 22.6%

For reference v0 reached POD 37.6% only at FAR 70.4% (CSI 19.87%), with a
threshold tuned on the report set itself.

---

### E8 — self-supervised pretraining (negative result)
Contrastive pretraining (SimCLR-style, NT-Xent, two meteorological views) on
30,000 unlabelled windows from the training period, 6 epochs, 18 minutes on one
RTX 4090, then 4 epochs of fine-tuning on the labelled set.

| model | dev AUC | benchmark AUC | benchmark PR-AUC | CSI at dev tau |
|---|---|---|---|---|
| supervised CNN | 0.7695 | 0.7211 | 0.3757 | — |
| SSL fine-tuned CNN | 0.7550 | 0.7208 | 0.3393 | — |
| ensemble: supervised CNN + GBDT | 0.7757 | 0.7329 | 0.3835 | 27.5% |
| ensemble: SSL CNN + GBDT | 0.7713 | 0.7364 | 0.3735 | 26.5% |

The SSL branch changes benchmark AUC by +0.0035 (95% CI of the benchmark AUC
is about +/-0.017) and lowers PR-AUC and CSI. Verdict: no measurable gain at
this budget, so it is **not** used in the final model. This is reported as a
negative result rather than silently dropped: 30k unlabelled windows and 18
minutes of pretraining are far below what contrastive learning normally needs,
and the unlabelled pool (about 900k windows) is large enough to revisit if
compute becomes available.

### Final model (frozen)
0.75 x supervised CNN + 0.25 x physics-feature GBDT, probability calibrated by
isotonic regression fitted on dev. Benchmark: AUC 0.733, PR-AUC 0.384,
ECE 0.012, Brier 0.095, CSI 27.5% at tau = 0.28 (POD 37.5%, FAR 49.1%).

### E9 — final test (opened once)
200 test timestamps frozen with seed 909 before any model was evaluated on the
test period (splits_test_eval.json, sha256 24dd1972...), 61,200 windows,
14,880 positive (24.3%). Threshold, ensemble weight and calibration were all
fixed from dev.

| metric | value |
|---|---|
| ROC-AUC | 0.8016 |
| PR-AUC | 0.5771 |
| CSI at dev tau (0.280) | 43.1% |
| POD / FAR / POFD | 66.1% / 44.7% / 17.2% |
| ETS / HSS | 0.298 / 0.459 |
| ECE (raw -> calibrated) | 0.252 -> 0.021 |
| Brier (raw -> calibrated) | 0.208 -> 0.138 |
| at POFD <= 10% | POD 50.9%, CSI 38.8% |
| at POFD <= 20% | POD 69.5%, CSI 42.9% |

Per case: 12 of 13 cases have AUC 0.75-0.83 (CSI 35-49%); one case
(SCW2021-0077) is weak at AUC 0.659 / CSI 16.2%.

The test period is more active than the benchmark (24.3% vs 12.8% positives)
and easier (AUC 0.802 vs 0.733); both are reported. The benchmark contains
three near-random cases that the larger test period does not.

### E10 — real ERA5 environment features (mixed result)
Six monthly ERA5 files were downloaded (pressure levels t/u/v/q at 200/850 hPa
and single levels CAPE/2 m T/MSL). Seven features are sampled at each window
centre and nearest hour: CAPE, 2 m temperature, MSL, 200-850 hPa wind shear,
850 hPa specific humidity, 850 hPa and 200 hPa temperature.

| model | dev AUC | benchmark AUC | benchmark PR | benchmark CSI |
|---|---|---|---|---|
| GBDT, 51 features | 0.7341 | 0.7072 | 0.2893 | 20.5% |
| GBDT, 51 + 7 ERA5 | 0.7627 | 0.7366 | 0.3509 | 22.8% |
| ensemble, no ERA5 | 0.7757 | 0.7329 | 0.3835 | 27.5% |
| ensemble, with ERA5 | 0.7939 | 0.7520 | 0.3896 | 27.7% |

On the held-out test subset the picture changes:

| variant | test AUC | test PR | test CSI | test POD/FAR |
|---|---|---|---|---|
| no ERA5 | 0.8016 | 0.5771 | 43.1% | 66.1% / 44.7% |
| with ERA5 | 0.8018 | 0.5748 | 42.0% | 61.6% / 43.0% |

Verdict: ERA5 clearly helps on the May 1-13 benchmark (+0.029 AUC for the GBDT,
+0.019 for the ensemble) but not on the May 14-31 test period (+0.0002 AUC,
-1.1 CSI). The environment features are not included in the final model. This
is reported as a mixed result rather than cherry-picking the benchmark gain.

### E11 — resolution and seed ensemble
Two ideas were tested on the benchmark (dev-tuned threshold in every row):

| model | dev AUC | benchmark AUC | benchmark PR | benchmark CSI |
|---|---|---|---|---|
| 128 px, 1 seed + GBDT | 0.7757 | 0.7329 | 0.3835 | 27.5% |
| 128 px, 1 seed + flip TTA + GBDT | 0.7765 | 0.7332 | 0.3786 | 27.5% |
| 224 px, 1 seed + GBDT | 0.7882 | 0.7405 | 0.3826 | 29.4% |
| 128 px, 5 seeds + TTA + GBDT | 0.7915 | 0.7433 | 0.3907 | 30.1% |
| blend of 5-seed + 224 + GBDT | 0.7957 | 0.7441 | 0.3971 | 30.7% |

Findings:
* TTA alone does nothing (0.7329 -> 0.7332); the gain comes from seed averaging.
* Resolution helps a little (+0.012 AUC for a single model, +0.008 for the
  ensemble) but is not the bottleneck.
* Combining 224 with the 5-seed ensemble adds only 0.0008 AUC for twice the
  training and inference cost, so it was not adopted.

Final model: 128 px, **5 seeds x 4-flip TTA**, blended 0.9 CNN + 0.1 GBDT.
On the frozen test subset it scores AUC 0.8062, PR-AUC 0.5742, CSI 42.0%
(POD 74.1%, FAR 50.8%), ECE 0.025. The single-seed version scored 0.8016 /
43.1% — the 5-seed version ranks better but spends more false alarms at the
dev-tuned threshold, so both are reported.

### E12 — object-level localisation: window peaks → physical centroids → dual guidance

*(This entry was missing from the log and was reconstructed from the archived
metric JSONs on 2026-11-13. Every number below was read back
out of `experiments/final/*.json`, not copied from a summary document.)*

Why: a window-level score answers "is there a cell somewhere in this ~250 km
box", not "where is it". Forecasters need a coordinate.

Protocol: primary tolerance **30 km**, greedy nearest-neighbour spatial bipartite
matching on Haversine distance; tolerance sensitivity at 15/30/45/60/90/112/120 km;
two peak modes — `local_max` (local peaks only) and `all_active` (every
above-threshold window). Ground truth: **463** benchmark cells, **6,806** test
cells. v1 rows use `calib.json` (w_cnn = 0.9, tau = 0.27295); v2 rows use
`cinet_v2_ensemble_metrics.json` (w_cnn = 0.95, tau = 0.30979).

**Step 1 — v1 + window geometric centre** (`localization_metrics.json`)

| split | mode | preds | hits | POD | CSI | FAR | mean offset |
|---|---|---|---|---|---|---|---|
| benchmark | local_max | 180 | 8 | 1.73% | 1.26% | 95.56% | 21.94 km |
| benchmark | all_active | 1,121 | 42 | 9.07% | 2.72% | 96.25% | 21.49 km |
| test | local_max | 2,815 | 142 | 2.09% | 1.50% | 94.96% | 17.92 km |
| test | all_active | 22,700 | 1,022 | 15.02% | 3.59% | 95.50% | 19.49 km |

A 250 km window centre is a poor proxy for a 10 km cell: even with perfect
window-level discrimination the offset stays around 18-22 km.

**Step 2 — v1 + microphysical centroid** (`localization_physics_metrics.json`)

`physics = 0.5 * cold + 0.5 * cool`, where `cold` is the minimum BT13 pixel at
T-10 and `cool` is the weighted centroid of the top 10% cooling pixels
(dBT13 = BT13(T-40) - BT13(T-10)).

| split | mode | geom | cold | cool | physics |
|---|---|---|---|---|---|
| benchmark | local_max | 8 | 10 | 5 | **11** |
| benchmark | all_active | 42 | 28 | 43 | **59** |
| test | local_max | 142 | 143 | **201** | 185 |
| test | all_active | 1,022 | 712 | 1,087 | **1,127** |

Findings:
* The **cooling-rate** field carries more initiation signal than the cold top
  alone — the coldest pixel is usually the mature anvil, not the initiating core
  (`cold` alone actually *hurts* on test all_active: 712 vs 1,022).
* The 0.5/0.5 blend is **not universally best**: on test local_max pure `cool`
  gets 201 hits (POD 2.95%, CSI 2.13%) versus 185 for the blend, while on the
  benchmark the blend wins (11 vs 5). The weight was picked on a small split and
  should be re-scanned on dev.

**Step 3 — CINet_v2, four coordinate strategies**
(`cinet_v2_localization_dual_metrics.json`)

A = window geometric centre, B = neural heatmap peak, C = physics centroid,
D = dual fusion `alpha * nn + (1 - alpha) * physics`.

| split | mode | preds | A hits | B hits | C hits | D hits | D alpha | D mean offset |
|---|---|---|---|---|---|---|---|---|
| benchmark | local_max | 168 | 7 | 6 | 7 | **12** | 0.6 | 19.06 km |
| benchmark | all_active | 1,041 | 41 | 40 | 54 | **65** | 0.6 | 18.75 km |
| test | local_max | 2,524 | 147 | 150 | 158 | **187** | 0.4 | 19.18 km |
| test | all_active | 21,647 | 1,016 | 1,061 | 1,127 | **1,389** | 0.8 | 18.67 km |

Findings:
* The heatmap head is the **most precise** localiser (mean offset 17.35 km on
  test local_max, 11.13 km on benchmark local_max) but not the highest-recall
  one. Precision and recall trade off against each other here.
* Dual fusion maximises hits: +27.2% over geometric on test local_max, +36.7% on
  test all_active, with CSI up 27.8% / 38.6% respectively.

> **Protocol violation confirmed by the 2026-11-13 audit.** `alpha` is swept over
> `np.linspace(0, 1, 11)` *inside the per-split loop* and the value with the
> highest CSI **on that same split** is kept
> (`tools/eval_localization_dual.py:634, 688-711`). That is tuning on the frozen
> test set, which this project's own protocol forbids. **All strategy-D numbers
> above are an optimistic upper bound, not an unbiased generalisation estimate.**
> Fix: select alpha on dev, freeze it, then score benchmark/test once.

**Step 4 — coarse anvil gating kills true cells (negative result)**

`decaying_anvil_ablation` in the same file, applying a window-scale decaying-anvil
test to the strategy-D candidates:

| split | mode | ungated hits | gated hits | preds before → after | FAR change |
|---|---|---|---|---|---|
| benchmark | local_max | 12 | 5 | 168 → 60 | −1.19 points (better) |
| benchmark | all_active | 65 | 21 | 1,041 → 342 | +0.10 points (worse) |
| test | local_max | **187** | **56** | 2,524 → 787 | **+0.29 points (worse)** |
| test | all_active | 1,389 | 496 | 21,647 → 6,915 | −0.76 points (better) |

On test local_max the gate suppressed 1,606 candidates (68.72% of all false
alarms) but **destroyed 131 true cells — 70% of everything it had detected** —
and FAR still got worse. A window-scale microphysical test cannot separate a
decaying anvil from an initiating core embedded inside it.

This is the empirical reason the gating had to become **local**: the 32 x 32
targeted gate in E13 suppresses 1,708 false alerts while retaining **99.18%** of
true cells. (But see E14 P3: that gate tests neighbourhood extrema and suppresses
only 3.24% of candidates, so the high retention reflects a weak gate rather than
precise targeting. The direction was right; the implementation still needs work.)

---

### E13 — 稠密连续态势感知、局域物理门控与多级预警操作体系验证 (Operational Multi-Tier Evaluation & Interactive Panel v2)

#### 1. 科学动机与业务瓶颈 (Operational Motivation)
过去面向切片网格（112-stride / 56-stride）的二分类评估虽然给出了宏观区域风险（ROC-AUC 0.8062, CSI 42.0%），但无法满足精细化气象防灾减灾的核心诉求：
1. **单体定位模糊**：网格打分无法精确给出对流触发核心的具体经纬度坐标与微物理冲顶中心；
2. **成熟冷砧虚警高企**：视场内大量已脱离对流发展期的成熟/衰退积雨云云砧（Decaying Anvils）依然极度寒冷，极易诱发假警报，导致预报员虚警疲劳（FAR 高达 95%+）；
3. **缺乏业务分级响应机制**：单一决策门限无法兼顾“最大检出早期监视”与“高确信应急联动”的矛盾。

#### 2. 方法体系与工程突破 (Methodology & Architecture)
1. **全域 2D 汉宁连续概率场与对流核心关键点场重建**：
   采用 56 步长滑窗，在全域 $2001 \times 2101$ 像元上通过 2D 汉宁加权窗无缝融合 5 种子 CINet_v2 模型的后验概率与对流中心关键点热图，生成平滑连续的风险场 $P_{\text{full}}(y, x)$ 与关键点热图 $H_{\text{full}}(y, x)$。
2. **双流空间坐标融合 (Dual Coordinate Refinement)**：
   综合神经网络语义峰值 $(y_{\text{nn}}, x_{\text{nn}})$ 与局域 $32 \times 32$ 像元最冷红外云顶像元质心 $(y_{\text{phys}}, x_{\text{phys}})$（$\alpha = 0.5$），将单体中心定位平均误差控制在 17 km 左右。
3. **局域微物理云砧门控 (Targeted Decaying Anvil Gating)**：
   针对候选点实施局域时空微物理检验：若 30 分钟冲顶降温率 $\Delta BT_{13,\text{local}} > -0.5\text{ K}$（停止降温或回温）且穿透水汽泡差值 $(BT_{10} - BT_{13})_{\text{local}} < 0\text{ K}$（无对流泡冲破对流层顶），判定为衰退期残留冷砧并予以定点剔除。
4. **业务三级预警操作体系 (Three-Tier Operational Warning Hierarchy)**：
   - 🔴 **红级预警 (Level 1 - Extreme Warning)**：高置信度 ($\tau_{\text{obj}} \ge 0.60$) 且剧烈冲顶急剧降温 ($\Delta BT_{13,\text{local}} \le -2.0\text{ K}$)，对应处于对流层顶剧烈爆发期的强对流极值单体，作为防灾应急联动的最高级别令签；
   - 🟠 **橙级警报 (Level 2 - Severe Alert)**：业务推荐最佳 CSI 平衡点（测试集取 $\tau_{\text{obj}} = 0.70$），大幅削减预报员发报负荷并最大化威胁评分；
   - 🟡 **黄级关注 (Level 3 - Surveillance Watch)**：全量有效候选单体池（门控后保留），维持最大检出率，用于全域监视追踪。

#### 3. 客观检验实证战果 (30 km Bipartite Matching Benchmarks)

基于冻结测试集（200 个独立时次，6,806 个真实 CI 单体）与基准集（30 个固定时次，463 个真实 CI 单体），在 30 km 业务容差下执行严格的最近邻空间双边匹配，扫描发报门限 $\tau_{\text{obj}} \in [0.30 \sim 0.80]$：

##### 3.1 独立测试集 (Held-out Test, 200 Timestamps, 6,806 GT Objects) 门限敏感性扫描
| 发报门限 $\tau_{\text{obj}}$ | 预测单体数 (Preds) | 命中数 (Hits) | 检出率 (POD) | 空报率 (FAR) | 威胁评分 (CSI) | 查准率 (Precision) |
|---|---|---|---|---|---|---|
| 0.30 | 50,104 | 2,410 | 35.41% | 95.19% | 4.42% | 4.81% |
| 0.35 | 47,905 | 2,377 | 34.93% | 95.04% | 4.54% | 4.96% |
| 0.40 | 43,870 | 2,325 | 34.16% | 94.70% | 4.81% | 5.30% |
| 0.45 | 38,747 | 2,253 | 33.10% | 94.19% | 5.20% | 5.81% |
| 0.50 | 34,579 | 2,167 | 31.84% | 93.73% | 5.53% | 6.27% |
| 0.55 | 30,514 | 2,056 | 30.21% | 93.26% | 5.83% | 6.74% |
| 0.60 | 26,679 | 1,929 | 28.34% | 92.77% | 6.11% | 7.23% |
| 0.65 | 22,762 | 1,753 | 25.76% | 92.30% | 6.30% | 7.70% |
| **0.70 (最佳平衡)** | **18,567** | **1,540** | **22.63%** | **91.71%** | **6.46%** | **8.29%** |
| 0.75 | 14,075 | 1,247 | 18.32% | 91.14% | 6.35% | 8.86% |
| 0.80 | 9,420 | 938 | 13.78% | 90.04% | 6.14% | 9.96% |

- **单体级 PR-AUC (30 km)**: **0.0167**（远高于自然先验的随机基准）。
- **空间匹配距离**: 平均匹配偏距 17.48 km，中位数 17.68 km，第一四分位数 (P25) 仅 11.59 km，验证了 10~20 km 级别的极佳空间聚焦能力。

##### 3.2 业务三级预警体系对比评估详表
| 预警级别 | 触发判定规则 | 评测范围 | 预测单体数 | 命中数 | 检出率 (POD) | 空报率 (FAR) | 威胁评分 (CSI) | 业务定位与价值 |
|---|---|---|---|---|---|---|---|---|
| 🟡 **黄级关注 (Level 3)** | 全量有效候选单体 (门控后) | 基准集 (30时次) | 3,723 | 97 / 463 | 20.95% | 97.39% | 2.37% | 全域早期连续监视，最大化捕捉潜在对流泡 |
| 🟡 **黄级关注 (Level 3)** | 全量有效候选单体 (门控后) | 测试集 (200时次) | 51,058 | 2,427 / 6,806 | **35.66%** | 95.25% | 4.38% | 保持最大召回率，全面建档追踪 |
| 🟠 **橙级警报 (Level 2)** | 最佳平衡点 ($\tau_{\text{obj}}=0.60$) | 基准集 (30时次) | 812 | 53 / 463 | 11.45% | 93.47% | **4.34%** | **发报量剧降 78.2%**，CSI 相对提升 **+82.8%** |
| 🟠 **橙级警报 (Level 2)** | 最佳平衡点 ($\tau_{\text{obj}}=0.70$) | 测试集 (200时次) | 18,567 | 1,540 / 6,806 | 22.63% | **91.71%** | **6.46%** | **虚警负荷削减 63.6%**，CSI 相对提升 **+47.6%** |
| 🔴 **红级预警 (Level 1)** | $\tau_{\text{obj}} \ge 0.60$ & $\Delta BT_{13} \le -2\text{K}$ | 基准集 (30时次) | 788 | 51 / 463 | 11.02% | 93.53% | 4.25% | 锁定强冷冲顶爆发核心，高置信应急响应 |
| 🔴 **红级预警 (Level 1)** | $\tau_{\text{obj}} \ge 0.60$ & $\Delta BT_{13} \le -2\text{K}$ | 测试集 (200时次) | 26,212 | 1,913 / 6,806 | 28.11% | 92.70% | 6.15% | 强深厚对流精准发报，坚守防灾减灾底线 |
| 🔴 **红级严格 (Level 1 Strict)** | $\tau_{\text{obj}} \ge 0.70$ & $\Delta BT_{13} \le -2\text{K}$ | 测试集 (200时次) | 18,270 | 1,527 / 6,806 | 22.44% | 91.64% | **6.48%** | 极低虚警率，实现单体级预警最极致威胁得分 |

#### 4. 业务工作负荷优化效益 (Workload Reduction Analysis)
对比黄级全量监视清单与橙级业务推荐发报：
- **测试集发报量从 51,058 报压缩至 18,567 报**，直接为预报员**消除 63.64% 的无效虚警核查负荷**；
- **CSI 威胁得分从 4.38% 跃升至 6.46%**（相对增益达 **+47.60%**）；
- **基准集发报量从 3,723 报压缩至 812 报**（负荷削减 **78.19%**），CSI 从 2.37% 翻倍至 4.34%（相对增益 **+82.83%**）。

#### 5. 新一代预报员全域交互态势看板 (Forecaster Interactive Panel v2)
研发并交付了面向气象台预报员实战环境的现代化态势标图看板：
1. **五层深度解耦多源态势架构**：
   - **Layer 1**: 葵花8号原生红外 B13 (10.4 μm) 亮温真彩云图（190~300 K 伪彩增强）；
   - **Layer 2**: 2D 汉宁窗平滑全域连续 CI 风险概率场 $P_{\text{full}}(y, x)$（$0 \sim 100\%$ 平滑热区）；
   - **Layer 3**: CINet_v2 对流核心空间关键点热图场 $H_{\text{full}}(y, x)$（对流中心激活泡）；
   - **Layer 4**: 地面多普勒雷达 35+ dBZ 组合反射率实况真值（鲜艳青色闭合轮廓与半透明掩膜）；
   - **Layer 5**: 业务三级预警单体分布（🔴红 / 🟠橙 / 🟡黄矢量圆标）与局域微物理门控剔除点标记（✖️灰色叉标）。
2. **轻量自包含单文件交互体系** (`reports/panels/panel_v2.html`，3.18 MB)：
   - 纯前端 Canvas 原生流体级平移与滚轮自由无级缩放（0.2x ~ 25x）；
   - 各图层独立无级透明度滑动条与可见性多选控制器；
   - 单体快速分级筛选过滤按钮（全量 / 红级 / 橙级 / 黄级 / 门控 / 空间命中）；
   - **单体悬停 HUD 浮窗诊断**：光标轻触任意预报单体，毫秒级弹出微物理深度剖面卡片（经纬度坐标、CI 发生概率、CINet 得分、30 分钟冲顶降温率 $\Delta BT_{13}$、对流冷顶亮温 $BT_{13}$、水汽差 $BT_{10}-BT_{13}$、冰晶化相态 $BT_{11}-BT_{13}$ 及实况 30km 双边匹配雷达检出状态）；
   - 零外部 CDN 依赖，纯单文件即开即用。
3. **高保真静态对比图** (`reports/panels/panel_v2.png`，7.6 MB)：
   四联画出版级科研图件，完整印证 5 大图层的物理一致性与对流爆发捕捉能力。

#### 6. 核心产物归档清册
- **评测指标全量数据库**: `experiments/final/operational_tiers_metrics.json`
- **单体 PR 曲线与操作特性三联画**: `reports/figures/operational_tiers_pr_curve.png`
- **单时次全域稠密感知缓存**: `experiments/final/dense_sample_test_435.npz`
- **高保真静态对比图件**: `reports/panels/panel_v2.png`
- **新一代预报员态势看板**: `reports/panels/panel_v2.html`
- **评测执行工具脚本**: `tools/eval_operational_tiers.py`
- **看板构建工具脚本**: `tools/make_panel_v2.py`
- **稠密特征抽取工具**: `tools/extract_sample_dense.py`

#### 7. 复核警示 (Protocol caveat, 2026-11-13 audit)

本节所有"最佳平衡点"门限都是**在评测划分自身上选出来的**，不是 Dev 上冻结的：
`tools/eval_operational_tiers.py:320,324` 把橙级 `opt_tau` 按划分硬编码为
benchmark 0.60 / test 0.70，恰好等于各自门限扫描的最大 CSI 点；红级 0.60 与红级
严格 0.70 同为硬编码常量。因此 CSI 6.46% / 6.48% 与"发报量 −63.64%"应视为
**乐观上界**。黄级全量候选（无门限选择）与门控前后对照（同一候选池）不受影响。
修复方式：在 Dev 上选定 τ_obj 后冻结，再对 Benchmark/Test 只评分一次。

另：稠密推理使用**固定 α=0.5**（`tools/dense_inference_v2.py:237-239`），与
E12 中按划分扫描出的 α 不是同一套口径，需统一。

### E14 — 全链路代码审计 (Full-pipeline code audit, 2026-11-13)

A full read-through of the v2 pipeline (data → training → inference → dense
perception → gating → object-level evaluation → metrics), plus a byte-level
comparison of the local mirror against the server workspace. Findings below all
carry code evidence; the two quantitative claims were measured directly from the
frozen annotations, not estimated.

#### 14.1 Verified correct (no action needed)

| item | evidence |
|---|---|
| No T0 leakage | `ewci/data.py:130` reads `obs[oi-4 : oi]` — frames T-40..T-10 only |
| 41-channel construction | `ewci/tensors.py:46-48`: 28 standardised bands + 6 BTD/5 + 7 (last−first)/2 |
| Time-based split isolation | train from `annotations_stage1_bg`, dev/bench/test from `annotations_full` filtered by frozen uid manifests |
| Model selection on dev only | `train_cnn_v2.py:255,440-442` picks best.pt by AUC on the first 20 dev uids |
| Calibration / weight / window threshold on dev only | `fit_calibration.py:23-62`, `eval_v2_ensemble.py:152-156` |
| TTA heatmap un-flipping | `predict_cnn_v2.py:131-134` flips the heatmap back before averaging — coordinates stay canonical |
| Hann reconstruction normalisation | `dense_inference_v2.py:171-176` accumulates weights and divides — a true weighted average |
| Spherical matching | `eval_localization.py:262` uses Haversine on real lat/lon; no flat-earth 1.11 km/px approximation in the matching path |
| Bilateral unique greedy matching | `eval_localization.py:266-276` sorts all pairs by distance globally, then marks both sides used |
| Meteorological formulas | `ewci/metrics.py:29-38` POD/FAR/POFD/CSI/ETS/HSS/BIAS/Brier all correct; ETS random-hit expectation `(h+m)(h+f)/n` correct |
| Window-level PR-AUC | `ewci/metrics.py:50` uses sklearn `average_precision_score` (proper AP) |
| ECE | `ewci/metrics.py:83-97` 10 equal-width bins, count-weighted |
| All 5 seeds trained fully | `logs/train_v2_690_{0..4}.out`: 6/6 epochs each, ~37.5 min, the 60-min budget never triggered |
| Local ↔ server integrity | 47 shared files, **zero MD5 drift** |

#### 14.2 Confirmed defects

**P1 (HIGH) — the heatmap positive mask is empty for 60.1% of positive windows.**
`train_cnn_v2.py:152` uses `pos_mask = target.ge(0.99)`. With sigma = 2 the
Gaussian reaches 0.99 only within 0.284 px of a grid point, but bbox centres map
to arbitrary sub-pixel positions (`:88-89`, scale 28/224 = 0.125).
Measured over the frozen test annotations: of **14,880** positive windows,
**8,942 (60.1%)** have a peak below 0.99, so `pos_loss` is identically zero and
`num_pos = max(1, ...)` (`:164`) collapses to 1. Example: bbox `[65,15,78,26]` →
centre (8.9375, 2.5625) → peak 0.9759.
Consequence: the heatmap head is driven almost entirely by the negative term.
Training logs confirm the imbalance — epoch 6 loss CLS 0.486 vs HM 2.727 with
lambda_hm = 1.0, so the shared backbone is dominated by a near-pure regulariser.
This is consistent with strategy B (pure heatmap) scoring only 150 hits versus
187 for dual fusion.
Fix: anchor the positive term on each object's nearest grid point (or normalise
the target by its own peak), and normalise by the true number of anchors.

**P2 (HIGH) — 14.5% of positive objects are edge-truncated but still used as labels and heatmap targets.**
Measured on the frozen test annotations: of **28,517** positive objects,
**4,144 have `complete=False`** and **188 have `state_valid=False`**.
`ewci/data.py:96 has_ci` only checks `label == 1`; `train_cnn_v2.py:85-89` uses the
clipped `bbox_xyxy` centre, which is biased toward the window interior.
Consequence: classification positives include half-cells, heatmap supervision is
systematically offset, and the object-level ground truth contains truncated cells.
Fix: decide and document one convention — either filter `complete=False`, or
recover the true centre from `full_grid_bbox_xyxy` and flag truncation.

**P3 (MEDIUM-HIGH) — anvil gating uses neighbourhood extrema, so it barely fires.**
`dense_inference_v2.py:220` takes `nanmin` of the cooling field over the whole
32x32 window and `:224` takes `nanmax` of BTD. The gate (`local_physics_gating.py:65`)
rejects only when *no pixel anywhere in 1,024* is cooling by more than 0.5 K and
*no pixel* has BTD >= 0. Both extrema bias toward keeping candidates.
Consequence: only 1,708 / 52,766 = **3.24%** suppressed. The headline "99.18% true-cell
retention" is a by-product of a very weak gate, not evidence of precise targeting.
Fix: evaluate the candidate core (e.g. 5x5 around the peak) or use quantiles
instead of extrema, then re-measure suppression and retention.

**P4 (MEDIUM) — the candidate score is normalised per timestamp, so tau_score is not comparable across scenes.**
`dense_inference_v2.py:192-194`: `H_norm = H_full / H_full.max()`,
`score_map = P_full * H_norm`, thresholded at `tau_score = 0.10`.
Consequence: a scene with one strong peak suppresses everything else; a flat scene
passes many peaks. The "continuous risk field" is not comparable across timestamps.
Fix: normalise by a dev-calibrated constant, or combine P_full and H_full without
per-scene rescaling.

**P5 (MEDIUM) — NMS radius equals the evaluation tolerance, capping recall.**
`dense_inference_v2.py:247` suppresses candidates within **30 km**, and the
evaluation tolerance is also **30 km**. Two true cells 25 km apart can never both
be reported. Fix: use an NMS radius well below the tolerance (10-15 km) or scale
it to the object size.

**P6 (MEDIUM) — the object-level "PR-AUC 0.01674" is a truncated 11-point trapezoid area.**
`eval_operational_tiers.py:119-134` integrates precision over recall at 11
thresholds (0.30..0.80), i.e. recall only spans **13.78%..35.41%**. Everything
below and above that range is missing, and trapezoidal interpolation on a PR
curve generally overestimates. It is also a different estimator from the
window-level `average_precision_score`, so the two are not comparable.
Fix: rank all candidates by score and compute proper AP, or rename the metric to
"truncated PR area (recall 13.8-35.4%)".

**P7 (MEDIUM) — the "physics centroid" in the dense stage is a single coldest pixel.**
`dense_inference_v2.py:231` uses `nanargmin`, not a centroid. The v1
`eval_localization_physics.py` uses a genuine weighted centroid, so the two stages
use different definitions under the same name. A single pixel is noise-sensitive
and can shift the coordinate by up to 16 px (~17 km).
Fix: align the terminology in the docs and consider a cold-region weighted centroid.

**P8 (LOW-MEDIUM) — threshold search only covers the upper half of the probability range.**
`ewci/metrics.py:58` and `:72` build the grid from quantiles 0.5..0.9995. If the
max-CSI threshold lies below the median probability it can never be found — a real
risk at 14.2% dev prevalence. Fix: widen to 0.01..0.999.

**P9 (LOW-MEDIUM) — model selection uses only 20 dev timestamps and is very noisy.**
From `logs/train_v2_690_*.out`, seed 0 dev AUC per epoch: 0.7656, 0.7712, **0.6729**,
0.7940, 0.8098, 0.8016; seed 3 peaks at **epoch 1** (0.8103) and declines afterwards.
Best-epoch varies from 1 to 5 across seeds. Fix: select on all 100 dev timestamps,
or add smoothing / patience.

**P10 (LOW) — the "brightness-temperature jitter" is not a BT jitter.**
`train_cnn_v2.py:122-123` multiplies all 41 channels by 0.98..1.02. The 28 band
channels are already standardised (scaling an anomaly is not a BT offset), and a
real BT offset would leave the 6 BTD and 7 difference channels unchanged.
Fix: perturb raw BT before building the tensor, or rename it "multiplicative noise".

**P11 (LOW) — dead code and non-disableable switches.**
`ci_label` / `category` branches (`train_cnn_v2.py:82-83`, `eval_localization.py:78-79`,
`extract_sample_dense.py:258`) never fire: the real annotation objects only carry
`label`, `state_label`, `state_valid`, `complete`, `bbox_xyxy`, `full_grid_bbox_xyxy`.
`--tta` and `--balanced-sampling` are `store_true, default=True`, so neither can be
turned off — the corresponding ablations cannot be run with these scripts.
`ewci/data.py:134 split_by_timestamp` is unused (random-by-timestamp; a leakage trap
if anyone ever calls it).

**P12 (NEEDS CONFIRMATION) — provenance of the normalisation climatology.**
`ewci/tensors.py:19` loads `/data1/group3/DATASAT/dataset/norm_stats.npz` from the
shared read-only dataset. Nothing in this repository records which split it was
computed on. If it includes May / test-period data, the standardisation carries a
mild leakage. Must be confirmed with the data provider; it cannot be fixed from
inside this repo.

#### 14.3 Suggested repair order

1. P1 + P2 (supervision signal quality — likely the largest single gain available,
   and cheap: no new architecture, just retrain).
2. The protocol breach already recorded in §4.6 (alpha / tau_obj on dev only).
3. P3 + P4 + P5 (candidate generation and gating — these set the recall ceiling).
4. P6 + P7 (metric and terminology honesty — documentation-level, do immediately).
5. P8 + P9 + P10 + P11 (search range, model selection, augmentation semantics, cleanup).
6. P12 (ask the data provider).

### E15 — 监督信号与评测协议修复后的重训重评 (Repair, retrain and honest re-evaluation, 2026-11-16)

Stage 1+2 of the E14 repair plan were implemented together (at the user's
request), then v2 was retrained and the whole chain re-run. **Attribution is
therefore mixed between the supervision fix and the protocol fix**; the two are
separated below wherever the artifacts allow it.

#### 15.1 What changed in code

| item | change |
|---|---|
| P1 | `_build_heatmap` forces the nearest grid point to exactly 1.0, so `pos_mask = target.ge(0.99)` fires and `num_pos` equals the real anchor count |
| P2 | anchors use the true centre from `full_grid_bbox_xyxy − (col, row)`, and are placed only when that centre lies inside the window; `state_valid=False` objects excluded |
| protocol | α swept on **dev only**, frozen per peak mode; τ_obj swept on **dev only**, frozen. `dense_inference_v2` and `eval_localization_dual` gained a dev split; `eval_v2_ensemble` now saves dev/benchmark prediction archives; `eval_operational_tiers` refuses to run without dev candidates |
| P6 | headline object metric is now average precision over **all** candidates ranked by score; the 11-point truncated area is kept as `pr_auc_truncated_11pt_legacy` |
| P7 | dense inference stores `y_phys/x_phys`; the `phys` ablation arm recomputes coordinates from them instead of copying the `dual` arm |
| P8 | threshold search grids widened from quantiles 0.5..0.9995 to 0.01..0.9995 |

Two bugs surfaced during the first full run and were fixed:
`annotations_stage1_bg/val_samples.jsonl` is a **symlink to an older 950-record
set** that misses 12 of the 100 dev timestamps (obs_index agrees on all 950
shared uids, so `annotations_full` with 1483 records is a safe superset);
and `eval_localization_dual` had two places where the dev split fell through to
the test branch. `benchmark_patches()` was deliberately left untouched because
it defines the frozen official 30-timestamp benchmark.

#### 15.2 Supervision fix verified before training

Measured on the frozen test annotations with the patched code:

| | before | after |
|---|---|---|
| positive windows with an empty heatmap mask | 8,942 / 14,880 (**60.1%**) | 852 (**5.7%**) |
| anchors total | — | 26,202 (mean 1.76 per window) |
| truncated-object centre bias | mean 3.3 km, max **62.2 km** | 0 (true centre used) |
| truncated objects whose centre is outside the window | 2,121 of 4,144 anchored at a clipped position | no anchor placed |

Training loss at epoch 6 (mean of five seeds): **HM 2.729 → 1.818 (−33.4%)**,
CLS 0.488 → 0.460 (−5.8%). All five seeds completed 6/6 epochs in ~40 min each
on five GPUs in parallel (47 min wall clock).

#### 15.3 Window-level results (protocol unchanged, τ still dev-only)

| split | metric | before | after | Δ |
|---|---|---|---|---|
| **test** | ROC-AUC | 0.8191 | **0.8298** | +0.0108 |
| test | PR-AUC | 0.5993 | **0.6142** | +0.0149 |
| test | CSI | 0.4362 | **0.4490** | +0.0128 |
| test | POD | 0.7456 | **0.7579** | +0.0124 |
| test | FAR | 0.4875 | **0.4758** | −0.0117 |
| dev | ROC-AUC | 0.7970 | 0.8022 | +0.0052 |
| benchmark | ROC-AUC | 0.7463 | 0.7480 | +0.0017 |
| benchmark | PR-AUC | 0.3851 | 0.3984 | +0.0133 |
| benchmark | CSI | 0.3035 | 0.2975 | **−0.0060** |
| benchmark | FAR | 0.5043 | 0.5251 | **+0.0207** |

All five test metrics improved. The benchmark is mixed (30 timestamps, noisy)
and is reported as such. Ensemble weight re-selected on dev: 0.95 → **0.98**;
τ_dev 0.30979 → **0.31435** (the wider P8 grid also contributes here).

#### 15.4 Object-level: the protocol fix makes the numbers honest, not better

α frozen on dev: **local_max 0.20**, **all_active 0.80** (previously swept on
each evaluation split: test local_max 0.4, test all_active 0.8).
τ_obj frozen on dev: **orange = red = red_strict = 0.65** (previously hardcoded
0.70 on test / 0.60 on benchmark, each that split's own best-CSI point).

**Dense perception + local gating, test** (no threshold selection involved, so
this isolates the model improvement):

| | before | after |
|---|---|---|
| raw candidates | 52,766 | 57,667 |
| pre-gating hits | 2,447 | **2,868** (+17.2%) |
| pre-gating POD | 0.3595 | **0.4214** |
| post-gating hits | 2,427 | **2,848** |
| post-gating POD | 0.3566 | **0.4185** |
| post-gating CSI | 0.0438 | **0.0475** |
| post-gating FAR | 0.9525 | **0.9491** |
| true-cell retention | 99.18% | 99.30% |

Benchmark post-gating: hits 97 → **121**, POD 0.2095 → **0.2613**,
CSI 0.0237 → **0.0346**, FAR 0.9739 → **0.9617**.

**Orange tier, test** — the honest dev-frozen operating point:

| | before (τ tuned on test) | after (τ frozen on dev) |
|---|---|---|
| τ_obj | 0.70 | 0.65 |
| alerts | 18,567 | 32,379 |
| hits | 1,540 | **2,353** (+53%) |
| POD | 0.2263 | **0.3457** |
| CSI | 0.0646 | 0.0639 |
| FAR | 0.9171 | 0.9273 |
| workload reduction vs yellow | 63.64% | **42.17%** |

The old headline (CSI 6.46%, FAR 91.71%, −63.6% workload) was an **optimistic
upper bound**. The honest point trades a slightly worse CSI/FAR for much higher
recall. Benchmark orange improved outright: hits 53 → 83, CSI 0.0434 → 0.0533.

**Dual-fusion strategy D** (α now frozen from dev):

| split / mode | before | after |
|---|---|---|
| test local_max | 187 hits, CSI 0.0205 | 174 hits, CSI 0.0187 |
| test all_active | 1,389 hits, CSI 0.0513 | **1,418 hits, CSI 0.0527** |
| benchmark local_max | 12 hits | 8 hits |
| benchmark all_active | 65 hits | 57 hits |

Meanwhile the α-free strategies improved on test local_max
(A 147→157, B 150→167, C 158→177), confirming the model itself got better.
**On test local_max, pure physics (177) now beats the dev-selected fusion (174)**
— the dev-optimal α = 0.20 does not transfer to that regime. This is a genuine
dev/test shift, recorded rather than hidden.

#### 15.5 P6 and P7 verified

**P6 — the AP jump is an estimator change, not a model gain:**

| combination | test AP |
|---|---|
| old model + old 11-point truncated estimator | 0.01674 |
| **new model + old truncated estimator** | **0.01494** |
| new model + new full-candidate AP | 0.04276 |

The recall coverage also moved (old 0.138–0.354, new 0.220–0.418), so the
truncated areas are not comparable across runs either. **0.04276 must not be
presented as a 2.6× model improvement.**

**P7 — the three ablation arms are now genuinely different** (they used to be
byte-identical for `pure_physics` and `dual_optimal`):

| split | pure_neural | dual | pure_physics |
|---|---|---|---|
| test | 2,790 hits / 16.86 km | **2,848 / 17.38 km** | 2,725 / 17.83 km |
| benchmark | 116 / 16.33 km | **121 / 16.86 km** | 117 / 17.80 km |

Dual fusion is the best of the three on both splits (+58 hits over neural and
+123 over physics on test), so the fusion genuinely helps — the previous
"identical numbers" were a broken control, not a degenerate fusion.

#### 15.6 Still open after this round

P3 (gating uses neighbourhood extrema; still only 1,679 / 57,667 = **2.9%**
suppressed), P4 (per-timestamp score normalisation), P5 (NMS radius = evaluation
tolerance), P9 (model selection on 20 dev timestamps — seed 3 still peaks at
epoch 1), P10, P11, P12. Also new: the dev τ sweep granularity (0.05) collapsed
orange/red/red_strict onto the same 0.65, so the red tiers now differ only by the
ΔBT₁₃ condition; a finer sweep is worth trying.

#### 15.7 Provenance

Old artifacts preserved in `experiments/archive_v2_original/` (97 MB, 44 files:
weights, predictions, caches, metric JSONs, panel). New authoritative numbers in
`experiments/final/`. Pipeline: `tools/run_fixed_pipeline.sh` (8 stages, five
GPUs, `FROM=n` to resume). Total wall clock: 47 min training + 21 min inference
and window evaluation + 18 min dense/object/panel, plus two debug iterations.

---

### E16 — 单体级虚警率根因诊断 (Object-level FAR root-cause diagnosis, 2026-11-17)

**Question.** After E15 the window-level FAR is 47.58% but the object-level FAR is
still 94.91% (yellow) / 92.73% (orange). Is that a training problem, a
post-processing problem, or an evaluation-protocol problem?

**Method.** `tools/diagnose_far.py` — read-only, CPU-only, trains nothing and
changes no operating point. It re-uses the cached dense candidates and the CIDS
annotations. Two equivalence checks make it comparable to the published numbers:

* the vectorised bilateral matcher reproduces `el.spatial_bipartite_matching`
  exactly on 8/8 sampled timestamps (hits, misses and false alarms all identical);
* its AP implementation reproduces `ot.compute_object_ap` to <1e-9 on all three
  splits (dev 0.02749, benchmark 0.02088, test 0.04276).

Output: `experiments/final/far_diagnosis.json`.

#### 16.1 The emission budget sets a hard FAR floor

| split | real objects | alerts after gating | alerts / object | FAR floor at **perfect** discrimination | actual FAR |
|---|---|---|---|---|---|
| dev | 1,871 | 12,766 | 6.82× | **0.8534** | 0.9560 |
| benchmark | 463 | 3,159 | 6.82× | **0.8534** | 0.9617 |
| test | 6,806 | 55,988 | **8.23×** | **0.8784** | 0.9491 |

Test issues 279.9 alerts per timestamp (median 276, p90 365, max 484) against 34.0
real objects. **Even a perfect discriminator at this emission rate could not get
FAR below 87.84%.** Of the observed 94.91%, roughly 88 points are "issued too
many" and only ~7 points are "picked the wrong ones". Reaching FAR 50% at the
current hit count would require cutting alerts by **89.8%**.

#### 16.2 But cutting the budget alone does not fix it either

Top-N alerts per timestamp, ranked by the current score, no ground truth used:

| N / timestamp | alerts | POD | FAR | CSI |
|---|---|---|---|---|
| 20 | 4,000 | 0.0717 | **0.8780** | 0.0473 |
| 34 | 6,800 | 0.1137 | 0.8862 | 0.0603 |
| 50 | 10,000 | 0.1556 | 0.8941 | 0.0673 |
| **100** | **20,000** | **0.2677** | **0.9089** | **0.0729** |
| 150 | 29,999 | 0.3373 | 0.9235 | 0.0665 |
| 280 | 50,497 | 0.4092 | 0.9448 | 0.0511 |
| current yellow (all) | 55,988 | 0.4185 | 0.9491 | 0.0475 |
| current orange τ=0.65 | 32,379 | 0.3457 | 0.9273 | 0.0639 |

**Oracle upper bound** (N = the true object count, i.e. cheating on the budget):
6,806 alerts → POD 0.1192, **FAR 0.8808**, CSI 0.0634. Even knowing exactly how
many cells exist, 88% of the emitted alerts are still wrong, because only 11.9%
of the top-ranked candidates are correct.

So the budget and the ranking are **jointly** binding. Neither lever alone
produces an operationally usable FAR.

The one free win: **top-100/timestamp beats the current orange tier on CSI
(0.0729 vs 0.0639) at 38% fewer alerts**, with no retraining. It does not
materially improve FAR.

#### 16.3 The ranking is weak, and the false alarms are not mislocalisations

TP-vs-FA discriminative AUC of every stored field (0.5 = no information):

| field | dev | benchmark | test |
|---|---|---|---|
| `prob` (window classifier) | 0.6818 | 0.7098 | 0.6845 |
| `score` = P × H/Hmax | 0.7056 | 0.7181 | 0.7010 |
| `hm_val` (keypoint head) | 0.6709 | 0.6749 | 0.6942 |
| `bt13_local` | 0.3283 | 0.3177 | 0.4073 |
| `delta_bt13_local` | 0.3983 | 0.4220 | 0.4177 |
| `b10_b13_local` | 0.6821 | 0.6958 | 0.6056 |
| `b11_b13_local` | 0.5581 | 0.5877 | 0.5246 |

Window-level ROC-AUC is 0.8298; the same confidence only reaches **0.70** when
asked "is this peak a real CI cell rather than some other cold cloud". The
physical fields are below 0.5, i.e. **inverted** — true cores are colder
(1−0.4073 = 0.593) and cool faster (1−0.4177 = 0.582) than false alarms, and
BTD₁₀₋₁₃ adds 0.606. They carry signal that the current score does not use.

AP over all candidates, and a null obtained by shuffling the score **within**
each timestamp (20 reps):

| ranking key | test AP | dev AP | benchmark AP |
|---|---|---|---|
| `score` (current) | 0.04276 | 0.02749 | 0.02088 |
| `prob` only | 0.04077 | 0.02399 | 0.01941 |
| `hm_val` only | 0.03749 | 0.02054 | 0.01629 |
| P × H, **no** per-timestamp normalisation | 0.04335 | 0.02509 | 0.02138 |
| −`bt13_local` | 0.02789 | 0.02151 | 0.01985 |
| −`delta_bt13_local` | 0.02803 | 0.01910 | 0.01317 |
| **NULL (shuffled)** | **0.02262 ± 0.00046** | 0.01464 ± 0.00061 | 0.01121 ± 0.00095 |

The model is **1.89× the random-ordering null** on test (1.88× dev, 1.86×
benchmark). Real signal, but far from a detector.

Distance from each false alarm to the **nearest** real CI object (test, 53,140 FAs):

| bucket | count | share |
|---|---|---|
| ≤30 km (lost the bilateral contest to a closer alert) | 240 | **0.5%** |
| 30–45 km | 2,590 | 4.9% |
| 45–60 km | 3,006 | 5.7% |
| 60–90 km | 6,609 | 12.4% |
| 90–150 km | 11,756 | 22.1% |
| **>150 km from any CI object** | **28,939** | **54.5%** |

**The majority of false alarms are not localisation errors and not duplicates.**
They are detections of something else — mature / decaying convection and cirrus
anvils — far from any initiating cell.

#### 16.4 Three plausible causes that the data refuted

Recorded explicitly because they were the working hypotheses before measuring:

1. **"P4 per-timestamp normalisation makes quiet timestamps emit alerts."**
   There are **zero** zero-GT timestamps in dev, benchmark or test — all sampled
   timestamps contain CI. Removing the normalisation changes test AP by
   +0.0006 and **dev AP by −0.0024**. P4 is a principled inconsistency, not a
   measurable FAR driver.
2. **"P5 NMS radius = 30 km leaves duplicate alerts that count as FAs."**
   Only 0.5% of FAs are within 30 km of a real object. Post-hoc dedup at 45 / 60 /
   90 km cuts alerts to 44,871 / 35,446 / 23,374 but FAR stays at 0.9476 /
   0.9474 / 0.9479 while POD falls proportionally. Spatial redundancy is not the
   problem.
3. **"The anvil gate is a big lever that is mis-tuned."** It fires on 2.91% of
   test candidates (dev 4.80%, benchmark 5.05%) and costs 20 true objects. The
   TP/FA medians do separate (ΔBT₁₃ −18.68 vs −15.84 K; BT₁₃ 244.82 vs 250.75 K),
   but the 32×32 neighbourhood **extrema** compress the separation so much that
   the AND rule almost never triggers. Fixing P3 will sharpen the inputs; it will
   not by itself remove 50,000 false alarms.

#### 16.5 Two real inconsistencies confirmed

**The dense path bypasses calibration.** `dense_inference_v2.py` thresholds the
raw Hann-blended CNN sigmoid at `tau_prob = 0.20`; the dev isotonic map and
`tau_dev = 0.31435` are never applied, and GBDT is not in the dense path at all.
On test, **88.69%** of windows have `p_raw ≥ 0.20` while only **35.15%** have
`p_cal ≥ tau_dev`. Mapping the dense peak probabilities through the window-level
isotonic curve (indicative only — the curve was fitted on the CNN+GBDT blend)
puts the median at 0.529 and leaves **74.88%** of candidates above `tau_dev`.
So the dense operating point is far more permissive than the window-level one.
Correcting it removes about a quarter of the alerts, not most of them.

**The keypoint head never commits.** Recovering Hmax from `hm_val·prob/score`
gives a median of **0.253** (p10 0.220, p90 0.315) on test, and its correlation
with the number of real objects is **−0.066**. A sigmoid head whose global
maximum sits at 0.25 is not producing confident keypoints; the 28×28 grid
(8 km/cell) with σ=2 cells and λ_hm=1.0 is under-resolved and under-weighted.

#### 16.6 Answer

**It is primarily a supervision / label problem, secondarily an emission-policy
problem, and only marginally a post-processing problem.**

The label is *"this 224×224 window contains at least one CI cell"*. A window
holding one 72-pixel initiating cell plus a large mature anvil satisfies it, so
the classifier is rewarded for responding to cold cloud texture in general. The
only object-level supervision is the coarse, under-confident heatmap head. That
is exactly the failure signature measured above: window AUC 0.83, object-level
TP/FA AUC 0.70, and 54.5% of false alarms more than 150 km from any CI cell.

#### 16.7 Ranked remedies (proposed, **not implemented**)

| # | action | expected effect | cost |
|---|---|---|---|
| 1 | **Pixel-level CI supervision.** The annotations already carry `contour_xy` / `contour_lonlat` per object, so a CI **mask head** can be trained instead of only a centre heatmap. Directly attacks the "fires on non-CI cloud" failure. | largest | retrain |
| 2 | **Sharper keypoint head**: 56×56 or 112×112 output, σ scaled down, λ_hm raised so Hmax actually approaches 1. | large | retrain |
| 3 | **Hard negatives from the full domain**: windows with cold cloud but no CI (the annotations label non-CI objects too). Training windows are currently 79% positive and built around CI cells. | large | retrain |
| 4 | **Learned object-level ranker** on `[prob, hm_val, bt13_local, delta_bt13_local, b10_b13_local, b11_b13_local]`, fitted on **dev only**. The physical fields carry independent signal (inverted AUC 0.59 / 0.58 / 0.61). | moderate, cheap | CPU minutes |
| 5 | **Fix P3**: compute the microphysics on the candidate **core** (≈5×5 or the peak pixel) instead of 32×32 extrema. Sharpens both the gate and remedy 4. | moderate | dense re-run ≈9 min, no retrain |
| 6 | **Emission policy**: per-timestamp top-N budget instead of a fixed τ_prob. Measured CSI 0.0729 at N=100 vs 0.0639 for the current orange tier, 38% fewer alerts. | small on FAR, real on CSI/workload | free |
| 7 | **Calibrate the dense path** on dev so τ_prob lives on the same scale as the window operating point. | removes ≈25% of alerts | small |

Remedies 1–3 need a retraining budget and are **not** started. 4–7 are cheap and
can be done without touching the CNN.

**Honest framing for the report.** At FAR ≈ 93% the "orange severe alert" tier is
not an operational warning product. The system should be described as a
**full-domain surveillance and ranking aid**, and the object-level FAR should be
reported next to the 8.23× emission ratio and the 0.8784 structural floor so the
number is not read as pure model failure.

#### 16.8 Two consistency bugs found and fixed during this check

* `tools/eval_operational_tiers.py` hard-coded **"+47.5% CSI"** in the panel (c)
  annotation of `reports/figures/operational_tiers_pr_curve.png`. The actual
  dev-frozen orange-tier gain is **+34.47%**. The published figure therefore
  displayed a stale pre-E15 number. Now computed from the tier results; figure
  re-rendered.
* Two comments still described the red tiers as fixed `tau >= 0.60` / `>= 0.70`;
  they are dev-selected. Comments corrected.
* `w_cnn` is **0.975** (the dev sweep grid is `np.linspace(0, 1, 41)`, step 0.025).
  Documents previously rounded it to 0.98; now stated precisely.

#### 16.9 GT-centre probe (a worry that turned out to be unfounded)

`load_ground_truth_objects` reads `bbox_xyxy` from `*_samples.jsonl`. At the
full-domain sample level the objects carry **only** `bbox_xyxy` — there is no
`full_grid_bbox_xyxy` (val 35,814 objects, test 74,514, zero with both). That
field exists only on patch-level records, where E15/P2 already handles it. So the
evaluation GT centres are full-grid centres and there is **no train/eval
coordinate mismatch**.

#### 16.10 Provenance

`experiments/final/far_diagnosis.json` (all tables above), produced by
`python tools/diagnose_far.py --splits dev,benchmark,test`. Inputs:
`dense_candidates_{dev,benchmark,test}.json`, `cinet_v2_{dev,bench,test}_predictions.npz`,
`annotations_full/{val,test}_samples.jsonl`. Runtime ≈3 min on CPU, no GPU.

---

## 4. Mistakes and corrections (kept on purpose)

1. Claimed 0.825 / +14 CSI from 17 features. Wrong: random split. Corrected
   to 0.709 under the time split.
2. Planned to label the test set from the dense raster mask. Wrong: the mask
   has large -1 (unknown) regions — only 4 of 18 annotation-positive windows
   were recovered on a sample timestamp. The test set is built from
   annotations_full instead.
3. First thought the test period had no annotations (test files in
   annotations_stage1_bg are empty). annotations_full/test_patches.jsonl has
   679,626 labelled windows.
4. First full ERA5 request was rejected by the CDS cost limit; downloads are
   now split into small chunks and limited to CAPE, 2 m temperature, MSL and
   t/u/v/q at 200/850 hPa.
5. **Coarse anvil gating (E12 step 4)** destroyed 70% of detected true cells on
   test local_max and made FAR worse. Replaced by the 32 x 32 local targeted
   gate (E13), which retains 99.18%.
6. **Operating points selected on the evaluation split (found 2026-11-13).**
   `eval_localization_dual.py:634,688-711` picks alpha by max CSI *per split*;
   `eval_operational_tiers.py:320,324` hardcodes the orange tau per split
   (benchmark 0.60, test 0.70), matching each split's own best-CSI sweep point,
   and the red / red-strict taus (0.60 / 0.70) are hardcoded constants too.
   This violates the frozen-test protocol. Not yet fixed: all affected
   object-level numbers must be treated as an optimistic upper bound until
   alpha and tau_obj are chosen on dev and frozen.
7. **Broken ablation control (found 2026-11-13).** In
   `local_physics_gating.py:73-78` the `dual` branch and the `else` branch meant
   to be `phys` are identical — both read the candidate's stored `lat/lon`.
   Hence `pure_physics` equals `dual_optimal` exactly in
   `dense_perception_metrics.json`. The dense fusion itself is fine
   (`dense_inference_v2.py:237-239`, fixed alpha = 0.5); the control is wrong.
8. **Misattributed results in the master memory (corrected 2026-11-13).** The
   an earlier internal results table labelled 12/463 and 187/6,806 as "v1 final
   ensemble"; both are CINet_v2 dual-fusion numbers. It also labelled 7/463 at
   23.15 km as the "v0 historical baseline"; that is CINet_v2 strategy A
   (geometric centre). The genuine v1 object-level numbers are 8/142 (geometric)
   and 11/185 (physics centroid). A "mean offset" column also carried a median
   (20.12 km instead of the 19.18 km mean).
9. **Two dev calibrations coexist.** `test_metrics_seed5.json` uses
   tau_dev = 0.26050 (dev AUC 0.79155) for the headline window metrics, while the
   persisted `calib.json` uses tau = 0.27295 (dev AUC 0.79180) for every
   localisation evaluation. Both are legitimate dev fits, but mixing them shifts
   CSI by about one point. Always state which file a number comes from.

---

## 5. Current status (as of 2026-11-13)

- **v1**: frozen protocol, calibrated, reproducible. Physics baseline AUC 0.707;
  5-seed ensemble benchmark 0.7433 / test 0.8062. Complete.
- **v2**: CINet_v2 (224 px, LEA + keypoint heatmap head), 5 seeds trained and
  archived (`experiments/p2_seed0..4/`). Window-level test AUC 0.8191, CSI
  43.62%, FAR 48.75%. Complete.
- **Object-level pipeline**: localisation, dual fusion, dense stride-56 Hann
  perception, 32 x 32 local gating and three-tier operational evaluation all run
  and archived. Delivered as `reports/panels/panel_v2.html` + `.png`.
  **Caveat: operating points were selected on the evaluation splits (see §4.6),
  so these numbers need a clean re-run.**
- **Test set**: opened once for the frozen v1/v2 window-level evaluation. It was
  subsequently used for alpha / tau_obj selection at the object level, which is
  a protocol breach to be repaired — not a reason to discard the data.
- **Semi-supervised stage**: only a 30k-window / 18-minute pilot was run (E8,
  negative). The full ~900k unlabelled pool is still unused.
- **True lead-time forecasting (T+10/T+20/T+30)**: not started. The current label
  is CI presence at T0, i.e. diagnosis.
- **Workspace**: development continues in
  `/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/`
  (git repository, full commit history kept). Other repositories under the same
  server account are read-only snapshots of earlier versions.

## 6. Reproduction

Run on the server, in the Qoder workspace:

~~~
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd /home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
export PYTHONPATH=.

# v1 physics baseline
python tools/build_features.py --split train --out experiments/p0_gbdt/features_train.npz
python tools/build_features.py --split dev   --out experiments/p0_gbdt/features_dev.npz
python tools/build_features.py --split val   --out experiments/p0_gbdt/features_val.npz
python tools/eval_frozen.py

# v1 deep ensemble and frozen final test
sbatch slurm/seeds_tta.slurm
python tools/fit_calibration.py
python tools/final_test5.py

# v2 training, inference, window-level ensemble
sbatch slurm/train_v2_parallel.slurm
bash tools/predict_all_seeds_v2.sh
python tools/eval_v2_ensemble.py

# v2 object level: localisation, dense perception, gating, tiers
python tools/eval_localization_dual.py
python tools/dense_inference_v2.py --split both --gpus 1,2,3,4 --stride 56
python tools/local_physics_gating.py
python tools/eval_operational_tiers.py

# forecaster panel v2
python tools/extract_sample_dense.py --uid test/435/SCW2021-0072
python tools/make_panel_v2.py
~~~

Check `df -h /` and `squeue -u group3` before submitting. No tmux/screen on the
cluster: long runs go through Slurm or `nohup`.
