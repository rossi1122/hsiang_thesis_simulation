# yang2024/xplane — 在 X-Plane 11 重現 Yang 2024（Phase 2–4）

學長的 `模擬程式/` **沒有被修改**；這裡的程式會從 `模擬程式/` 匯入 XPlaneConnect 與 YOLO 執行緒來用。
所以 `yang2024/` 資料夾要放在和 `模擬程式/` **同一層**。

## 和學長程式的差別
| | 學長 `模擬程式/main.py` | 這裡 `run_yang.py` |
|---|---|---|
| 任務 | 俯衝攻擊（SEARCH → PRE_DIVE → DIVE） | 追蹤盤旋（Yang 2024 切換式 MPC：J_g / J₂ / J₁） |
| 外迴路 | 影像誤差 PID 或 3DOF MPC | Yang 的影像式 MPC（和數值模擬 `yang2024/mpc.py` 同一份） |
| 轉向方式 | **方向舵**轉向、機翼保持水平 | **傾斜轉彎**：u_ψ → φ_cmd = atan(V·u_ψ/g) → 副翼；方向舵只做協調 |
| 雲台 | 學長自己的 PD + 穩定 + 俯衝鎖定 | Yang Eq. 9（前作 Eq. 18–20）：每週期把目標拉回中心 |
| 讀狀態 | 每圈分開讀約 30 次 | 每圈一次 `getDREFs` |
| 時間 | 每圈固定加 0.01 s（實際比真實慢一倍） | 用真實經過時間 |
| 目標特徵 | YOLO | `truth`：吉普車真值投影（Stage 3）；`yolo`：學長的 YOLO 執行緒（Stage 4） |

## 兩個版本（`--profile`）
三支程式（camera_check、step_test、run_yang）都可以加 `--profile`：

| | `--profile scaled`（預設，尺度版） | `--profile original`（論文原值） |
|---|---|---|
| V / H | 30 m/s / 300 m | 16 m/s / 100 m |
| 參數 | 依下表按比例換算，無因次量和論文相同 | 直接用 Table 1（k = 1） |
| 吉普車速度（V_c = 7） | 13.1 m/s | 7 m/s |
| 狀態 | SP80 已驗證可飛、內迴路已調好 | **要先用 step_test 確認 SP80 能飛 16 m/s**；需要改的增益寫在 `profiles.original.autopilot` |

結果資料夾名稱會帶版本，例如 `20261117_160000_original_yang_switch_truth_paper`。
共用設定（相機、連線、自駕儀增益）寫在 `xp_params.yaml` 上方；各版本只寫不一樣的部分。

## 參數縮放（`xp_params.yaml`）
尺度版照比例放大，保持所有無因次量和論文相同：

| | 論文（= original 版） | X-Plane 尺度版（V = 30 m/s、H = 300 m） |
|---|---|---|
| 高度 H / 空速 V_t | 100 m / 16 m/s | 300 m / 30 m/s |
| 盤旋半徑 R_d / 感知範圍 R_M / ΔR | 50 / 100 / 10 m | 150 / 300 / 30 m |
| u_max | 0.5 rad/s（坡度 39°） | 0.3125 rad/s（坡度 44°） |
| Δt / 預測時域 | 0.05 s / 0.75 s | 0.08 s / 1.2 s |
| 目標速度 V_c = 5 / 7 / 9 | 5 / 7 / 9 m/s | 9.4 / 13.1 / 16.9 m/s |
| 焦距 f | 537 px | 1662.8 px（比較 RMSE 時乘 537/f） |

## 檔案
| 檔案 | 作用 |
|---|---|
| `xp_params.yaml` | 尺度、相機、連線、內迴路增益、起始位置、階躍測試設定 |
| `config.py` | 讀設定並把論文參數換算到 X-Plane 尺度 |
| `xp_link.py` | 和 X-Plane 溝通：座標換算、讀狀態與吉普車、送控制與雲台、擺放飛機 |
| `autopilot.py` | 傾斜轉彎內迴路 + 定高 + 定速 |
| `camera.py` | 相機內參、像素換算、吉普車真值投影 |
| `camera_check.py` | 檢查相機模型（截圖上畫出預測位置） |
| `step_test.py` | 開迴路階躍測試：τ、延遲、u_max 做不做得到、高度損失 |
| `run_yang.py` | Yang 2024 閉迴路 |
| `fake_xplane.py` | 假的 X-Plane，只給離線測試（`--dry-run`） |

所有指令都在 repo 根目錄（有 `yang2024/` 和 `模擬程式/` 的那層）、`dive_sim` 環境下執行。
結果存在 `results/xplane/<日期_時間>_<名稱>/`：`log.csv`、`summary.json`、圖。

## 即時顯示視窗（`--show`、`--record`）
類似學長 main.py 的顯示視窗：綠色十字圈 = 真值投影、紅框 = YOLO、左上 = MPC 模式（J1 綠／J2 黃／Jg 紅）、n1、R、u_ψ、坡度、YOLO 狀態。
```
python -m yang2024.xplane.run_yang --controller switch --measure yolo --yolo-model <best.pt> --vc 7 --show --record
```
- 視窗預設開在 x = 1920（主螢幕右邊的第二個螢幕）；第二個螢幕在左邊時加 `--show-x -1280`。
- **YOLO 是截主螢幕**：視窗不能放在 X-Plane 那個螢幕上，否則 YOLO 會看到視窗。
- `--record` 把畫面存成結果資料夾裡的 `overlay.mp4`（約 8 fps），可以直接放進簡報。
- 顯示在獨立執行緒，不影響 50 Hz 控制迴圈。
