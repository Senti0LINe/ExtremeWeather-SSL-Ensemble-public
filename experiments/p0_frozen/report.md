# Frozen-protocol baseline (physics features + LightGBM)

train 2021-03-01..04-30 | dev 30600 windows / 100 timestamps | benchmark 9180 windows

| split | n | pos | AUC | PR-AUC | CSI@tau | POD | FAR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| dev | 30600 | 4356 | 0.7341 | 0.3375 | 24.5% | 51.7% | 68.3% | 0.2128 | 0.277 |
| benchmark | 9180 | 1175 | 0.7072 | 0.2893 | 20.5% | 44.5% | 72.5% | 0.2095 | 0.276 |

Benchmark AUC 95% CI: [0.690, 0.724] | CSI 95% CI: [18.9%, 22.1%]

At FAR<=20% on the benchmark: POD 0.1%, CSI 0.1%

Reference v0: AUC 0.6745 | PR-AUC 0.2474 | CSI 19.87%% @ tau 0.64

## Per case (benchmark)
- SCW2021-0063: n=1224 AUC=0.596 CSI=0.059
- SCW2021-0064: n=1224 AUC=0.760 CSI=0.233
- SCW2021-0065: n=918 AUC=0.786 CSI=0.412
- SCW2021-0066: n=1224 AUC=0.694 CSI=0.223
- SCW2021-0067: n=1224 AUC=0.799 CSI=0.277
- SCW2021-0068: n=1224 AUC=0.604 CSI=0.125
- SCW2021-0069: n=918 AUC=0.559 CSI=0.023
- SCW2021-0070: n=1224 AUC=0.660 CSI=0.196