"""路口右轉實驗的兩個主要指標（用 run_yang 的 log.csv 離線計算，不需要 X-Plane）。

指標一：轉彎期間，距離 R 跑出追蹤範圍 [R_d, R_M]（尺度版 150–300 m）的時間
指標二：最長連續跟丟時間（meas_valid = 0；閘門擋掉的偵測也算跟丟）

轉彎時間窗：吉普車開始減速 → 回到原速後再加 --after 秒；吉普車停車（< 1 m/s）之後的資料不採用。
吉普車速度由真值位置 (jx, jy) 差分得到。

用法（每個實驗資料夾裡要有 log.csv）：
    python -m yang2024.xplane.analyze_turn results/xplane/20261122_173941_* results/xplane/20261122_2005* \
        results/xplane/20261122_2019* --out results/xplane/baseline_turn_metrics
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_log(path):
    path = Path(path)
    if path.is_dir():
        path = path / "log.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0].keys():
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        cols[k] = np.array(vals)
    return cols


def jeep_speed(t, jx, jy, smooth_s=1.0):
    """真值位置差分得到的吉普車速度，再做 smooth_s 秒的移動平均。"""
    v = np.full_like(t, np.nan)
    dt = np.diff(t)
    ok = dt > 1e-6
    sp = np.full(len(dt), np.nan)
    sp[ok] = np.hypot(np.diff(jx), np.diff(jy))[ok] / dt[ok]
    v[1:] = sp
    v[0] = v[1] if len(v) > 1 else np.nan
    out = np.empty_like(v)
    for i in range(len(t)):
        m = (t >= t[i] - smooth_s / 2) & (t <= t[i] + smooth_s / 2) & np.isfinite(v)
        out[i] = np.mean(v[m]) if m.any() else np.nan
    return out


def find_turn_window(t, v, v_cruise=None, drop=0.85, recover=0.95, after=5.0, stop_v=1.0):
    """回傳 (t_brake, t_recover, t_end, t_stop)。找不到減速則 t_brake = nan。"""
    finite = np.isfinite(v)
    if v_cruise is None:
        v_cruise = float(np.nanpercentile(v[finite], 90))
    i_brake = None
    for i in range(len(t)):
        if finite[i] and v[i] < drop * v_cruise and np.nanmax(v[: i + 1]) > recover * v_cruise:
            i_brake = i
            break
    # 停車：最後一次速度 < stop_v 之前、且之前有在開
    t_stop = math.inf
    moving = False
    for i in range(len(t)):
        if finite[i] and v[i] > 5.0:
            moving = True
        if moving and finite[i] and v[i] < stop_v and (i_brake is None or t[i] > t[i_brake] + 3.0):
            t_stop = float(t[i])
            break
    if i_brake is None:
        return math.nan, math.nan, float(min(t[-1], t_stop)), t_stop
    # 往前找真正開始減速的點（速度開始掉到 recover 以下）
    j = i_brake
    while j > 0 and finite[j - 1] and v[j - 1] < recover * v_cruise:
        j -= 1
    t_brake = float(t[j])
    # 減速之後第一次回到原速 = 出彎（停車前）
    t_recover = math.nan
    for i in range(i_brake, len(t)):
        if t[i] >= t_stop:
            break
        if finite[i] and v[i] >= recover * v_cruise:
            t_recover = float(t[i])
            break
    end = (t_recover if math.isfinite(t_recover) else t[-1]) + after
    return t_brake, t_recover, float(min(end, t[-1], t_stop)), t_stop


def time_where(t, mask):
    """mask 為 True 的總時間（每一筆資料代表到下一筆的時間）。"""
    if len(t) < 2:
        return 0.0
    dt = np.diff(t, append=t[-1])
    return float(np.sum(dt[mask]))


def longest_run(t, mask):
    """mask 連續為 True 的最長時間。"""
    best, start = 0.0, None
    for i in range(len(t)):
        if mask[i] and start is None:
            start = t[i]
        if start is not None and (not mask[i] or i == len(t) - 1):
            end = t[i]
            best = max(best, end - start)
            start = None
    return float(best)


def turn_metrics(log, band=(150.0, 300.0), after=5.0, name=""):
    t = log["t"]
    v = jeep_speed(t, log["jx"], log["jy"])
    t_brake, t_rec, t_end, t_stop = find_turn_window(t, v, after=after)
    R = log["R_truth"]
    valid = log["meas_valid"]
    roll = log["roll"]
    use = t <= min(t_stop, t[-1])
    win = use & (t >= t_brake) & (t <= t_end) if math.isfinite(t_brake) else np.zeros_like(use)
    Rw = R[win]
    out = dict(
        name=name,
        t_brake=t_brake, t_recover=t_rec, t_end=t_end, t_stop=None if math.isinf(t_stop) else t_stop,
        jeep_v_min=float(np.nanmin(v[win])) if win.any() else math.nan,
        # 指標一
        out_of_band_s=time_where(t[win], (Rw < band[0]) | (Rw > band[1])) if win.any() else math.nan,
        below_band_s=time_where(t[win], Rw < band[0]) if win.any() else math.nan,
        above_band_s=time_where(t[win], Rw > band[1]) if win.any() else math.nan,
        R_min=float(np.nanmin(Rw)) if win.any() else math.nan,
        t_R_min=float(t[win][np.nanargmin(Rw)]) if win.any() else math.nan,
        R_max_after_min=math.nan,
        # 指標二
        longest_loss_turn_s=longest_run(t[win], valid[win] < 0.5) if win.any() else math.nan,
        longest_loss_run_s=longest_run(t[use], valid[use] < 0.5),
        detection_turn=float(np.mean(valid[win] > 0.5)) if win.any() else math.nan,
        # 輔助
        max_bank_turn_deg=float(np.nanmax(np.abs(roll[win]))) if win.any() else math.nan,
    )
    if win.any():
        after_min = win & (t >= out["t_R_min"])
        out["R_max_after_min"] = float(np.nanmax(R[after_min]))
    if "pan_rate_act" in log:
        pr = log["pan_rate_act"][use]
        out["gimbal_pan_rate_max_dps"] = float(np.nanmax(np.abs(np.degrees(pr)))) if np.isfinite(pr).any() else math.nan
    return out, dict(t=t, v=v, R=R, valid=valid, roll=roll, use=use, win=win)


def _fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def summary_table(results):
    keys = [("t_brake", "吉普車開始減速 (s)"), ("jeep_v_min", "彎中最低車速 (m/s)"),
            ("out_of_band_s", "★指標一：跑出 [150, 300] m 的時間 (s)"), ("below_band_s", "　其中 R < 150 m (s)"),
            ("above_band_s", "　其中 R > 300 m (s)"), ("R_min", "最近距離 (m)"),
            ("R_max_after_min", "之後被甩到最遠 (m)"),
            ("longest_loss_turn_s", "★指標二：轉彎期間最長跟丟 (s)"), ("longest_loss_run_s", "整段實驗最長跟丟 (s)"),
            ("detection_turn", "轉彎期間偵測率"), ("max_bank_turn_deg", "轉彎期間最大坡度 (°)")]
    head = "| 項目 | " + " | ".join(r["name"] for r in results) + " | 平均 ± 標準差 |"
    lines = [head, "|" + "---|" * (len(results) + 2)]
    for k, label in keys:
        vals = [r.get(k, math.nan) for r in results]
        if k == "detection_turn":
            cells = [_fmt(100 * x) + "%" if x is not None and math.isfinite(x) else "—" for x in vals]
        else:
            cells = [_fmt(x) for x in vals]
        arr = np.array([x for x in vals if x is not None and math.isfinite(x)], float)
        if k == "t_brake" or len(arr) == 0:
            ms = ""
        elif k == "detection_turn":
            ms = f"{100 * arr.mean():.1f} ± {100 * arr.std(ddof=1) if len(arr) > 1 else 0:.1f}%"
        else:
            ms = f"{arr.mean():.1f} ± {arr.std(ddof=1) if len(arr) > 1 else 0:.1f}"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {ms} |")
    return "\n".join(lines)


def plot(results, series, out_png, band=(150.0, 300.0)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for r, s in zip(results, series):
        tb = r["t_brake"] if math.isfinite(r["t_brake"]) else 0.0
        tt = s["t"] - tb
        u = s["use"]
        ax[0].plot(tt[u], s["v"][u], label=r["name"])
        ax[1].plot(tt[u], s["R"][u], label=r["name"])
        ax[2].plot(tt[u], np.abs(s["roll"][u]), label=r["name"])
    ax[1].axhspan(band[0], band[1], color="tab:green", alpha=0.1, label="tracking band")
    for a, yl in zip(ax, ["jeep speed (m/s)", "horizontal distance R (m)", "|bank| (deg)"]):
        a.set_ylabel(yl)
        a.axvline(0, color="k", lw=0.8, ls="--")
        a.grid(alpha=0.3)
    ax[0].legend(loc="lower left", fontsize=8)
    ax[2].set_xlabel("time from jeep braking (s)")
    ax[0].set_title("Right-turn scenario: aligned at jeep braking")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="實驗資料夾（含 log.csv）或 log.csv 路徑")
    ap.add_argument("--names", nargs="*", help="每次實驗的名稱（預設 Run 1, Run 2, ...）")
    ap.add_argument("--band", nargs=2, type=float, default=[150.0, 300.0], metavar=("R_LOW", "R_HIGH"))
    ap.add_argument("--after", type=float, default=5.0, help="吉普車回到原速後再多算幾秒（預設 5）")
    ap.add_argument("--out", default="turn_metrics", help="輸出資料夾")
    a = ap.parse_args(argv)
    names = a.names or [f"Run {i + 1}" for i in range(len(a.runs))]
    results, series = [], []
    for p, n in zip(a.runs, names):
        r, s = turn_metrics(load_log(p), band=tuple(a.band), after=a.after, name=n)
        r["source"] = str(p)
        results.append(r)
        series.append(s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = summary_table(results)
    (out / "turn_metrics.md").write_text(table + "\n", encoding="utf-8")
    (out / "turn_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(results, series, out / "turn_compare.png", band=tuple(a.band))
    print(table)
    print(f"\n已存：{out / 'turn_metrics.md'}、{out / 'turn_metrics.json'}、{out / 'turn_compare.png'}")


if __name__ == "__main__":
    main()
