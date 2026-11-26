"""路口右轉實驗的兩個主要指標（用 run_yang 的 log.csv 離線計算，不需要 X-Plane）。

指標一：轉彎期間，距離 R 跑出追蹤範圍 [R_d, R_M]（尺度版 150–300 m）的時間
        （R < 150 全算；R > 300 只算飛機已進入範圍之後，減速時還在追近的那段不算）
指標二：吉普車開始減速之後（到資料結束）的最長連續跟丟時間（meas_valid = 0；閘門擋掉的也算）

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


def jeep_speed(t, jx, jy, smooth_s=2.0):
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


def find_turn_window(t, v, v_cruise=None, deep=0.6, recover=0.9, back=0.95, after=5.0, stop_v=1.0):
    """回傳 (t_brake, t_recover, t_end, t_stop)。找不到轉彎則 t_brake = nan。

    X-Plane 吉普車的速度雜訊很大（20 m/s 命令時 20–28 m/s 跳動），所以：
    1. 巡航速度 = 行駛中速度的中位數；
    2. 速度掉到巡航的 deep（60%）以下才算「真的在減速轉彎」；
    3. 再往回找速度開始低於巡航 back（95%）的時間 = 開始減速；
    4. 之後第一次回到巡航的 recover（90%）= 出彎；停車（< stop_v）之前沒回來就當作沒轉彎。
    """
    finite = np.isfinite(v)
    moving = finite & (v > 5.0)
    if not moving.any():
        return math.nan, math.nan, float(t[-1]), math.inf
    if v_cruise is None:
        v_cruise = float(np.median(v[moving]))
    seen_cruise = False
    i_deep = None
    for i in range(len(t)):
        if not finite[i]:
            continue
        if v[i] > recover * v_cruise:
            seen_cruise = True
        if seen_cruise and v[i] < deep * v_cruise:
            i_deep = i
            break
    t_recover, t_brake, t_stop = math.nan, math.nan, math.inf
    i_rec = None
    if i_deep is not None:
        for i in range(i_deep, len(t)):
            if not finite[i]:
                continue
            if v[i] < stop_v:
                break                       # 直接停車，不是轉彎
            if v[i] >= recover * v_cruise:
                i_rec = i
                break
    if i_rec is not None:
        j = i_deep
        while j > 0 and (not finite[j - 1] or v[j - 1] < back * v_cruise):
            j -= 1
        t_brake, t_recover = float(t[j]), float(t[i_rec])
    start = i_rec if i_rec is not None else 0
    for i in range(start, len(t)):
        if finite[i] and v[i] < stop_v and (i_rec is not None or np.nanmax(v[: i + 1]) > 5.0):
            t_stop = float(t[i])
            break
    if not math.isfinite(t_brake):
        return math.nan, math.nan, float(min(t[-1], t_stop)), t_stop
    end = t_recover + after
    return t_brake, t_recover, float(min(end, t[-1], t_stop)), t_stop


def time_where(t, mask):
    """mask 為 True 的總時間（每一筆資料代表到下一筆的時間）。"""
    if len(t) < 2:
        return 0.0
    dt = np.diff(t, append=t[-1])
    return float(np.sum(dt[np.asarray(mask, bool)]))


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
    t = log["t"] - log["t"][0]          # log 的 t 可能是電腦時鐘 → 改成從 0 開始
    v = jeep_speed(t, log["jx"], log["jy"])
    t_brake, t_rec, t_end, t_stop = find_turn_window(t, v, after=after)
    R = log["R_truth"]
    valid = log["meas_valid"]
    roll = log["roll"]
    use = t <= min(t_stop, t[-1])
    win = use & (t >= t_brake) & (t <= t_end) if math.isfinite(t_brake) else np.zeros_like(use)
    # 減速之後到資料結束（吉普車停車或 log 結束）：大坡度補救、跟丟多半發生在這段
    post = use & (t >= t_brake) if math.isfinite(t_brake) else np.zeros_like(use)
    Rw = R[win]
    # 「太遠」只算飛機進入範圍之後：減速時飛機可能還在追近（例：減速時 R = 365 m），那不是被甩出去
    t_enter = math.nan
    if win.any():
        inside = np.where(Rw <= band[1])[0]
        t_enter = float(t[win][inside[0]]) if len(inside) else math.inf
    above = win & (R > band[1]) & (t >= t_enter) if win.any() else win
    below = win & (R < band[0])
    out = dict(
        name=name,
        t_brake=t_brake, t_recover=t_rec, t_end=t_end, t_stop=None if math.isinf(t_stop) else t_stop,
        jeep_v_min=float(np.nanmin(v[win])) if win.any() else math.nan,
        R_at_brake=float(R[win][0]) if win.any() else math.nan,
        post_brake_s=float(t[post][-1] - t[post][0]) if post.any() else math.nan,
        # 指標一：轉彎時間窗內，進入範圍後又跑出 [R_low, R_high] 的時間
        below_band_s=time_where(t, below) if win.any() else math.nan,
        above_band_s=time_where(t, above) if win.any() else math.nan,
        out_of_band_s=math.nan,
        R_min=float(np.nanmin(Rw)) if win.any() else math.nan,
        t_R_min=float(t[win][np.nanargmin(Rw)]) if win.any() else math.nan,
        R_max_after_min=math.nan,
        # 指標二：減速之後（到資料結束）的最長連續跟丟
        longest_loss_after_brake_s=longest_run(t[post], valid[post] < 0.5) if post.any() else math.nan,
        detection_after_brake=float(np.mean(valid[post] > 0.5)) if post.any() else math.nan,
        longest_loss_turn_s=longest_run(t[win], valid[win] < 0.5) if win.any() else math.nan,
        longest_loss_run_s=longest_run(t[use], valid[use] < 0.5),
        detection_turn=float(np.mean(valid[win] > 0.5)) if win.any() else math.nan,
        # 輔助
        max_bank_turn_deg=float(np.nanmax(np.abs(roll[win]))) if win.any() else math.nan,
        max_bank_after_brake_deg=float(np.nanmax(np.abs(roll[post]))) if post.any() else math.nan,
    )
    if win.any():
        out["out_of_band_s"] = out["below_band_s"] + out["above_band_s"]
    if win.any():
        after_min = post & (t >= out["t_R_min"])
        out["R_max_after_min"] = float(np.nanmax(R[after_min]))
    if "pan_rate_act" in log:
        pr = log["pan_rate_act"][use]
        out["gimbal_pan_rate_max_dps"] = float(np.nanmax(np.abs(np.degrees(pr)))) if np.isfinite(pr).any() else math.nan
    return out, dict(t=t, v=v, R=R, valid=valid, roll=roll, use=use, win=win)


def _fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def summary_table(results):
    keys = [("t_brake", "吉普車開始減速 (s)"), ("R_at_brake", "開始減速時的距離 R (m)"),
            ("jeep_v_min", "彎中最低車速 (m/s)"), ("post_brake_s", "減速後的資料長度 (s)"),
            ("out_of_band_s", "★指標一：轉彎時跑出 [150, 300] m 的時間 (s)"), ("below_band_s", "　其中 R < 150 m (s)"),
            ("above_band_s", "　其中進入後又 > 300 m (s)"), ("R_min", "最近距離 (m)"),
            ("R_max_after_min", "之後被甩到最遠 (m)"),
            ("longest_loss_after_brake_s", "★指標二：減速後最長連續跟丟 (s)"),
            ("detection_after_brake", "減速後偵測率"), ("longest_loss_run_s", "整段實驗最長跟丟 (s)"),
            ("max_bank_after_brake_deg", "減速後最大坡度 (°)")]
    head = "| 項目 | " + " | ".join(r["name"] for r in results) + " | 平均 ± 標準差 |"
    lines = [head, "|" + "---|" * (len(results) + 2)]
    for k, label in keys:
        vals = [r.get(k, math.nan) for r in results]
        if k.startswith("detection"):
            cells = [_fmt(100 * x) + "%" if x is not None and math.isfinite(x) else "—" for x in vals]
        else:
            cells = [_fmt(x) for x in vals]
        arr = np.array([x for x in vals if x is not None and math.isfinite(x)], float)
        if k == "t_brake" or len(arr) == 0:
            ms = ""
        elif k.startswith("detection"):
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
    print(table)
    png = out / "turn_compare.png"
    try:
        plot(results, series, png, band=tuple(a.band))
    except OSError as e:
        # Windows：舊的 png 被看圖程式開著（或被同步軟體鎖住）時無法覆蓋 → 改存新檔名
        import time
        png = out / f"turn_compare_{time.strftime('%H%M%S')}.png"
        print(f"\n⚠️ 無法覆蓋 turn_compare.png（{e}）；可能是圖片還開著。改存成 {png.name}")
        plot(results, series, png, band=tuple(a.band))
    print(f"\n已存：{out / 'turn_metrics.md'}、{out / 'turn_metrics.json'}、{png}")


if __name__ == "__main__":
    main()
