"""CLAUDE.md 第 6 節要求記錄的指標。"""
import numpy as np


def summarize(log, params):
    dt, R_M, u_max = params["dt"], params["R_M"], params["u_max"]
    err = np.hypot(log["u2"], log["v2"])
    out = log["R"] > R_M
    # 連續超出 R_M 的段落
    episodes, run = [], 0
    for flag in out:
        if flag:
            run += 1
        elif run:
            episodes.append(run * dt)
            run = 0
    if run:
        episodes.append(run * dt)
    solve = log["solve_time"][log["solve_time"] > 0]
    return {
        "rmse_s2_px": float(np.sqrt(np.mean(err**2))),
        "rmse_s2_norm_x537": float(np.sqrt(np.mean(err**2)) * 537.0 / params["f"]),
        "R_max_m": float(log["R"].max()),
        "R_min_m": float(log["R"].min()),
        "time_outside_RM_s": float(out.sum() * dt),
        "outside_episodes": len(episodes),
        "longest_outside_s": float(max(episodes)) if episodes else 0.0,
        "u_sat_ratio": float(np.mean(np.abs(log["u_psi"]) >= 0.999 * u_max)),
        "solve_time_mean_ms": float(solve.mean() * 1e3) if solve.size else 0.0,
        "solve_time_max_ms": float(solve.max() * 1e3) if solve.size else 0.0,
        "solve_failures": int(np.sum(~log["solve_ok"].astype(bool))),
        "out_of_fov_ratio": float(np.mean(~log["in_fov"].astype(bool))),
        "duration_s": float(log["t"][-1] + dt),
    }
