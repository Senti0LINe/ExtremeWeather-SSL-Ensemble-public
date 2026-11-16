# P0 physics-feature GBDT baseline

Train: 2021-03-01..04-30 (all windows). Val: official 30 May timestamps.
Threshold tuned on 6 held-out timestamps, reported on the other 24.

| Split | Windows | ROC-AUC | PR-AUC | CSI | POD | FAR | Brier |
|---|---|---|---|---|---|---|---|
| held-out 24 ts (primary) | 7344 | 0.706 | 0.306 | 20.6% | 52.7% | 74.7% | 0.212 |
| official 30 ts | 9180 | 0.705 | 0.287 | 20.0% | 54.2% | 75.9% | 0.217 |

At FAR<=10% (tau=0.992): POD 0.1% | CSI 0.1%

Reference - legacy v0 ensemble: ROC-AUC 0.6745 | PR-AUC 0.2474 | CSI 19.87% (tau 0.64)

## Per case (held-out)
- SCW2021-0063: n=918 AUC=0.584 CSI=0.062
- SCW2021-0064: n=918 AUC=0.766 CSI=0.236
- SCW2021-0065: n=918 AUC=0.785 CSI=0.406
- SCW2021-0066: n=1224 AUC=0.695 CSI=0.225
- SCW2021-0067: n=918 AUC=0.795 CSI=0.270
- SCW2021-0068: n=918 AUC=0.586 CSI=0.122
- SCW2021-0069: n=918 AUC=0.562 CSI=0.054
- SCW2021-0070: n=612 AUC=0.634 CSI=0.191