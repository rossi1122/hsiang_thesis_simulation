"""Phase 1 第 2 步：從目標後方遠距離起始（回答老師疑慮 (1)：目標在遠處時演算法怎麼運作）。

情境
- 飛機在目標正後方 d = 300 / 500 m，航向與目標相同（+x），雲台一開始對準目標。
- 目標速度 V_c = 5 / 10 / 15 m/s（V_c / V_t = 0.31 / 0.63 / 0.94）。
- 目標運動：直線；以及 d = 300、V_c = 5 / 10 時的兩種轉彎，
  在預估追上時間 + 30 s 後開始：
    circle：等速圓周，角速度 0.05 rad/s（論文 Fig. 14 的設定）
    sharp ：90° 左急轉（0.3 rad/s，約 5.2 s），之後直線
- 控制器：切換式 MPC（提出方法）、P1,soft（不切換）、LSM（前作）。

用法（在 repo 根目錄）：
    python -m yang2024.far_start            # 全部 30 組，4 個行程平行
    python -m yang2024.far_start --quick    # 只跑 d = 300 直線，快速檢查
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import REPO_DIR, load_params  # noqa: E402
from .sim import simulate  # noqa: E402

CONTROLLERS = ("switch", "p1soft", "lsm")
LABELS = {"switch": "proposed (switching MPC)", "p1soft": "P1,soft", "lsm": "LSM"}
COLORS = {"switch": "#2a9d8f", "p1soft": "#e76f51", "lsm": "#7b6fd6"}

TURN_RATE = {"circle": 0.05, "sharp": 0.3}
TURN_DURATION = {"circle": None, "sharp": (np.pi / 2) / 0.3}


def catch_up_time(d, Vc, p):
    """直線追上到 R_M 以內的粗略時間（飛機直飛向目標）。"""
    return (d - p["R_M"]) / (p["V_t"] - Vc)


def scenarios(p, quick=False):
    out = []
    for d in (300.0, 500.0):
        for Vc in (5.0, 10.0, 15.0):
            if quick and d != 300.0:
                continue
            tc = catch_up_time(d, Vc, p)
            out.append(dict(name=f"d{int(d)}_v{int(Vc)}_straight", d=d, Vc=Vc,
                            motion="straight", T=round(tc + 150.0), t_turn=None))
    if not quick:
        for Vc in (5.0, 10.0):
            tc = catch_up_time(300.0, Vc, p)
            for motion in ("circle", "sharp"):
                out.append(dict(name=f"d300_v{int(Vc)}_{motion}", d=300.0, Vc=Vc,
                                motion=motion, T=round(tc + 180.0), t_turn=round(tc + 30.0)))
    return out


class TurnProfile:
    """目標在 t_turn 之後開始轉彎（可 pickle，給多行程用）。"""

    def __init__(self, Vc, motion, t_turn):
        self.Vc, self.motion, self.t_turn = Vc, motion, t_turn

    def __call__(self, t):
        if self.motion == "straight" or t < self.t_turn:
            return self.Vc, 0.0
        dur = TURN_DURATION[self.motion]
        if dur is not None and t >= self.t_turn + dur:
            return self.Vc, 0.0
        return self.Vc, TURN_RATE[self.motion]


def far_metrics(log, p):
    """遠距離情境的指標：分成「追近」與「進入 R_M 之後的追蹤」兩段。"""
    dt, R_M, f = p["dt"], p["R_M"], p["f"]
    R, t = log["R"], log["t"]
    inside = R <= R_M
    entered = bool(inside.any())
    k0 = int(np.argmax(inside)) if entered else len(R)
    track = slice(k0, None)
    err = np.hypot(log["u2"], log["v2"])

    def episodes(mask):
        eps, run = [], 0
        for flag in mask:
            if flag:
                run += 1
            elif run:
                eps.append(run * dt)
                run = 0
        if run:
            eps.append(run * dt)
        return eps

    out_after = ~inside[track]
    eps = episodes(out_after) if entered else []
    modes, counts = np.unique(log["mode"], return_counts=True)
    solve = log["solve_time"][log["solve_time"] > 0]
    return {
        "entered_RM": entered,
        "t_enter_s": float(t[k0]) if entered else None,
        "R_final_m": float(R[-1]),
        "R_min_m": float(R.min()),
        "rmse_s2_track_px": float(np.sqrt(np.mean(err[track] ** 2)) * 537.0 / f) if entered else None,
        "R_max_after_enter_m": float(R[track].max()) if entered else None,
        "outside_ratio_after_enter": float(out_after.mean()) if entered else None,
        "exits_after_enter": len(eps),
        "longest_outside_after_enter_s": float(max(eps)) if eps else 0.0,
        "u_sat_ratio": float(np.mean(np.abs(log["u_psi"]) >= 0.999 * p["u_max"])),
        "ideal_invalid_ratio": float(1.0 - np.mean(log["ideal_valid"])),
        "out_of_fov_ratio": float(np.mean(~log["in_fov"].astype(bool))),
        "mode_time_s": {str(m): round(float(c * dt), 1) for m, c in zip(modes, counts)},
        "solve_time_mean_ms": float(solve.mean() * 1e3) if solve.size else 0.0,
        "solve_time_max_ms": float(solve.max() * 1e3) if solve.size else 0.0,
        "solve_failures": int(np.sum(~log["solve_ok"].astype(bool))),
    }


_MPC = None


def _job(args):
    """單一模擬（在子行程裡跑；每個行程只建一次 MPC 求解器）。"""
    global _MPC
    sc, ctrl, p = args
    if ctrl != "lsm" and _MPC is None:
        from .mpc import YangMPC
        _MPC = YangMPC(p)
    profile = TurnProfile(sc["Vc"], sc["motion"], sc["t_turn"])
    t0 = time.time()
    log = simulate(p, ctrl, Vc=sc["Vc"], T=sc["T"], target_profile=profile,
                   uav_init=(-sc["d"], 0.0), uav_heading=0.0,
                   target_init=(0.0, 0.0), target_heading=0.0,
                   mpc=_MPC if ctrl != "lsm" else None)
    m = far_metrics(log, p)
    m["wall_time_s"] = round(time.time() - t0, 1)
    keep = {k: log[k] for k in ("t", "x", "y", "tx", "ty", "R", "u_psi", "mode", "u2", "v2")}
    return sc["name"], ctrl, m, keep


def _fmt(v, nd=1):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_tables(results, scs, out_dir):
    cols = ["entered_RM", "t_enter_s", "R_max_after_enter_m", "outside_ratio_after_enter",
            "longest_outside_after_enter_s", "rmse_s2_track_px", "R_final_m", "u_sat_ratio",
            "ideal_invalid_ratio", "solve_time_mean_ms", "solve_failures"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "d_m", "Vc", "motion", "controller"] + cols)
        for sc in scs:
            for ctrl in CONTROLLERS:
                m = results[sc["name"]][ctrl]
                w.writerow([sc["name"], sc["d"], sc["Vc"], sc["motion"], ctrl] + [m[c] for c in cols])
    lines = ["| 情境 | 控制器 | 進入 R_M | 進入時間 (s) | 之後 R 最大 (m) | 之後在外比例 | 之後最長在外 (s) "
             "| 追蹤 RMSE (px) | 結束時 R (m) | u 飽和 | s2 無效 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc in scs:
        for ctrl in CONTROLLERS:
            m = results[sc["name"]][ctrl]
            lines.append(
                f"| {sc['name']} | {ctrl} | {_fmt(m['entered_RM'])} | {_fmt(m['t_enter_s'])} "
                f"| {_fmt(m['R_max_after_enter_m'])} | {_fmt(m['outside_ratio_after_enter'], 2)} "
                f"| {_fmt(m['longest_outside_after_enter_s'])} | {_fmt(m['rmse_s2_track_px'])} "
                f"| {_fmt(m['R_final_m'])} | {_fmt(m['u_sat_ratio'], 2)} | {_fmt(m['ideal_invalid_ratio'], 3)} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_grid(logs, scs, p, path, title):
    n = len(scs)
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False)
    for ax, sc in zip(axes.ravel(), scs):
        for ctrl in CONTROLLERS:
            lg = logs[sc["name"]][ctrl]
            ax.plot(lg["t"], lg["R"], color=COLORS[ctrl], lw=1.2, label=LABELS[ctrl])
        ax.axhline(p["R_M"], color="k", ls="--", lw=1)
        if sc["t_turn"] is not None:
            ax.axvline(sc["t_turn"], color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_title(sc["name"], fontsize=10)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("R (m)")
        ax.grid(alpha=0.3, which="both")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trajectories(logs, scs, path, title):
    n = len(scs)
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows), squeeze=False)
    for ax, sc in zip(axes.ravel(), scs):
        first = None
        for ctrl in CONTROLLERS:
            lg = logs[sc["name"]][ctrl]
            ax.plot(lg["x"], lg["y"], color=COLORS[ctrl], lw=1.0, label=LABELS[ctrl])
            first = first or lg
        ax.plot(first["tx"], first["ty"], color="k", lw=2, label="target")
        ax.plot(first["x"][0], first["y"][0], "k^")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(sc["name"], fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    p = load_params()
    scs = scenarios(p, quick=args.quick)
    jobs = [(sc, ctrl, p) for sc in scs for ctrl in CONTROLLERS]
    out_dir = REPO_DIR / "results" / f"{datetime.now():%Y%m%d}_yang2024_far_start{'_quick' if args.quick else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {sc["name"]: {} for sc in scs}
    logs = {sc["name"]: {} for sc in scs}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, ctrl, m, lg in ex.map(_job, jobs):
            results[name][ctrl] = m
            logs[name][ctrl] = lg
            print(f"{name:22s} {ctrl:7s} 進入R_M={_fmt(m['entered_RM'])} t={_fmt(m['t_enter_s'])} "
                  f"R_final={m['R_final_m']:.0f} ({m['wall_time_s']} s)")
    print(f"總計 {time.time() - t0:.0f} s")

    straight = [s for s in scs if s["motion"] == "straight"]
    turning = [s for s in scs if s["motion"] != "straight"]
    plot_grid(logs, straight, p, out_dir / "R_straight.png",
              "Far start, straight target: horizontal distance R (log scale, dashed = R_M)")
    plot_trajectories(logs, straight, out_dir / "traj_straight.png", "Far start, straight target: trajectories")
    if turning:
        plot_grid(logs, turning, p, out_dir / "R_turning.png",
                  "Far start (300 m), turning target: R (dotted = turn starts)")
        plot_trajectories(logs, turning, out_dir / "traj_turning.png", "Far start (300 m), turning target")

    write_tables(results, scs, out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"date": datetime.now().isoformat(timespec="seconds"), "scenarios": scs,
                   "results": results, "params": p}, fh, ensure_ascii=False, indent=2)
    print(f"結果存在 {out_dir.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
