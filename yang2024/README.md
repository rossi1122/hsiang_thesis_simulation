# yang2024 — Yang et al. 2024 (Drones 8(4):136) 數值重現

Phase 1 第 1 步：不接 X-Plane，用 Python 重現論文 Section 4 的數值模擬（Fig. 4–6）。
和論文不同、或論文沒寫清楚而自行決定的地方，全部記在 `docs/reproduction_log.md`。

## 安裝
```bash
pip install casadi numpy matplotlib pyyaml pytest
```

## 執行（在 repo 根目錄）
```bash
python -m pytest yang2024/tests -q          # 自我檢查（座標慣例、Eq. 8、MPC 限制）
python -m yang2024.experiments fig4         # LSM vs 提出方法，V_c = 7 m/s（約 10 秒）
python -m yang2024.experiments fig5         # P1,soft vs 提出方法，V_c = 9 m/s
python -m yang2024.experiments fig6         # 目標複雜運動，500 s
python -m yang2024.experiments all
```
- 遠距離起始（Phase 1 第 2 步，老師疑慮 (1)）：`python -m yang2024.far_start`（30 組，4 行程平行約 1 分鐘；`--quick` 只跑 300 m 直線）。
  結果在 `results/<日期>_yang2024_far_start/`：`summary.md` 表格、R 對時間圖、軌跡圖。
- 影像運動拆解（Phase 1 第 3 步，RQ1）：`python -m yang2024.motion_budget`（約 1.5 分鐘），結果在 `results/<日期>_yang2024_motion_budget/`。
- 結果存到 `results/<日期>_yang2024_<實驗>/`：比較圖、`metrics.json`（RMSE、R 最大值、超出 R_M 的時間、求解時間、u_ψ 飽和比例）、實際使用的參數。
- 覆寫參數：`--set switch_mapping=proof hinge_eps=0`；縮短模擬：`--T 30`。

## 檔案
| 檔案 | 內容 | 論文 |
|---|---|---|
| `params.yaml` | 所有參數；論文沒寫的標「補」 | Table 1 |
| `geometry.py` | 座標慣例、投影、Ideal State 轉換、垂直相機、深度、影像模型 | Eq. 4–8、15、20、27、54 |
| `models.py` | 飛機 unicycle、雲台與雲台律、目標運動 | Eq. 1–3、9（前作 Eq. 18–20） |
| `baselines.py` | LSM 對照組 | 前作 Yang 2021 JIRS Eq. 13–17 |
| `mpc.py` | J1/J2/Jg 三個 CasADi + IPOPT 求解器、warm start、切換計數器 | Eq. 10–14、22–43 |
| `sim.py` | 閉迴路模擬 | Algorithm 1 |
| `metrics.py` | CLAUDE.md 第 6 節要求的指標 | — |
| `experiments.py` | Fig. 4–6 的實驗設定與輸出 | Section 4 |
| `far_start.py` | 從目標後方 300/500 m 起始、V_c = 5/10/15、目標轉彎 | （論文沒有，回答老師疑慮 (1)） |
| `motion_budget.py` | Eq. 8 預測誤差拆成 ④ 目標、② 偏航延遲；0.75 s 預測誤差；真實相機運動預算 | （論文沒有，回答 RQ1） |
| `tests/` | 用數值微分驗證 Eq. 8、Eq. 54、穩態盤旋等 | — |

## 座標慣例（細節見 `geometry.py` 開頭）
世界 x、y 水平、z 朝上，航向逆時針為正（u_ψ < 0 是右轉）；機體 FLU；
pan 往左為正、tilt 往下為負，Ideal State 為 θ_p = -π/2（右側）、θ_t = -α；
影像 u 往右、v 往下。用這組慣例推導出的影像運動和論文 Eq. 8 完全相同。
