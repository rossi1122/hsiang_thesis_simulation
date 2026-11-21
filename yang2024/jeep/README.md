# 目標吉普車情境腳本

`jeep_turn.py`：吉普車右轉 90° 的測試與情境（在 jeep 筆電的 X-Plane 執行），沿用學長的
`xplane_sim_jeep/jeep_controller.py`（原檔不修改）。

| 模式 | 意義 | 速度 |
|---|---|---|
| `curve` | 大半徑曲線（預設 200 m） | 不變 |
| `corner` | 一般路口轉彎（預設 20 m） | 彎中受側向加速度限制 v ≤ √(a_lat·r)（預設 a_lat = 3 m/s² ≈ 0.3 g → 7.7 m/s），進彎減速、出彎加速 |
| `noslow` | 測試：小半徑但不減速 | 不變（看 X-Plane 的車會不會打滑、翻車或轉不過去） |

轉彎掉速是物理造成的：側向加速度 = v²/r 不能超過輪胎摩擦（約 0.8 g），一般駕駛只用到約 0.3 g。
20 m/s 不減速要轉 90°，半徑至少要 51 m（物理極限）、舒適要 136 m；路口半徑 12–20 m 時只能 6–12 m/s。

```
python yang2024/jeep/jeep_turn.py --mode curve  --speed 20 --radius 200
python yang2024/jeep/jeep_turn.py --mode corner --speed 20 --radius 20
python yang2024/jeep/jeep_turn.py --mode noslow --speed 20 --radius 20
```
log 在執行目錄的 `jeep_turn_logs/`，含 `wall_time`（和飛機 log 對時）、階段、命令、速度、航向、車身滾轉、側向加速度。
結束時會印出：開始轉彎速度、最低速度、實際轉彎半徑、最大側向加速度、最大車身滾轉。
