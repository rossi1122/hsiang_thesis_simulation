"""Phase 2：開迴路階躍測試（不需要吉普車）。

先定高定速平飛 hold_s 秒，再依序送 u_psi 階躍命令（xp_params.yaml 的 step_test.steps_frac × u_max），量測：
  - 實際航向變化率 ψ̇ 對命令的一階反應：時間常數 τ、延遲 t_d、穩態增益（Phase 4 的 ② 要用）
  - 最大命令 u_max 做不做得到（穩態增益是否接近 1）、需要多少坡度
  - 轉彎時高度、空速掉多少（內迴路增益要不要調）

用法（飛機 PC，repo 根目錄）：
    python -m yang2024.xplane.step_test               # 從目前位置往上擺到 H，開始測試
    python -m yang2024.xplane.step_test --dry-run     # 用假的 X-Plane 測試程式
    python -m yang2024.xplane.step_test --profile original   # 論文原值版 V 16 m/s、H 100 m
"""
import argparse
import math
from pathlib import Path

import numpy as np

from .autopilot import BankToTurnAutopilot
from .common import add_profile_arg, Recorder, output_dir, save_json, setup


def step_commands(cfg, u_max):
    """u_psi 命令 [rad/s]：有 steps（絕對值）就用，否則 steps_frac × u_max。"""
    if "steps" in cfg:
        return [float(c) for c in cfg["steps"]]
    return [round(float(c) * u_max, 4) for c in cfg["steps_frac"]]


def schedule(cfg, u_max):
    """回傳 [(開始時間, 結束時間, 命令)]。"""
    seg, t = [(0.0, cfg["hold_s"], 0.0)], cfg["hold_s"]
    for c in step_commands(cfg, u_max):
        seg.append((t, t + cfg["step_s"], float(c)))
        t += cfg["step_s"]
    return seg


def fit_first_order(t, y, y0, cmd):
    """y(t) = y0 + A·(1 − exp(−(t − t_d)/τ))（t > t_d）。網格搜尋 τ、t_d，A 用最小平方。"""
    best = None
    for td in np.linspace(0.0, 1.0, 21):
        for tau in np.geomspace(0.05, 5.0, 60):
            b = np.where(t > td, 1.0 - np.exp(-(t - td) / tau), 0.0)
            denom = float(b @ b)
            if denom < 1e-9:
                continue
            A = float(b @ (y - y0)) / denom
            sse = float(np.sum((y - y0 - A * b) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tau, td, A)
    _, tau, td, A = best
    gain = A / (cmd - y0) if abs(cmd - y0) > 1e-6 else float("nan")
    return {"tau_s": tau, "delay_s": td, "gain": gain, "final_rate": y0 + A}


def analyze(rec, segs, params):
    t = rec.column("t")
    t = t - t[0]
    # 用協調轉彎的航向變化率 g·tanφ/V 擬合：由滾轉角算，比用機體角速度 (P, Q, R) 算的平滑很多
    rate, roll = rec.column("psi_rate_coord"), rec.column("roll")
    agl, spd = rec.column("agl"), rec.column("airspeed")
    results = []
    for i, (a, b, cmd) in enumerate(segs[1:], start=1):
        m = (t >= a) & (t < b)
        if m.sum() < 5:
            continue
        prev = segs[i - 1][2]
        y0 = float(np.nanmean(rate[(t >= a - 1.0) & (t < a)])) if np.any((t >= a - 1.0) & (t < a)) else prev
        r = {"t_start": a, "cmd": cmd, "prev_cmd": prev,
             "bank_mean_last5s_deg": float(np.nanmean(roll[m & (t >= b - 5)])),
             "agl_max_err_m": float(np.nanmax(np.abs(agl[m] - params["H"]))),
             "airspeed_min": float(np.nanmin(spd[m])), "airspeed_max": float(np.nanmax(spd[m]))}
        if abs(cmd - prev) > 0.05:
            r.update(fit_first_order(t[m] - a, rate[m], y0, cmd))
        results.append(r)
    taus = [r["tau_s"] for r in results if "tau_s" in r]
    gains = [r["gain"] for r in results if "gain" in r and abs(r["cmd"]) >= 0.999 * params["u_max"]]
    return {
        "steps": results,
        "tau_median_s": float(np.median(taus)) if taus else None,
        "delay_median_s": float(np.median([r["delay_s"] for r in results if "delay_s" in r])) if taus else None,
        "gain_at_u_max": gains,
        "hold_agl_rms_err_m": float(np.sqrt(np.nanmean((agl[t < segs[0][1]] - params["H"]) ** 2))),
        "hold_airspeed_rms_err": float(np.sqrt(np.nanmean((spd[t < segs[0][1]] - params["V_t"]) ** 2))),
        # 升降舵抖動：平飛時的標準差。2026-11-17 第一次測 0.68（全推全拉振盪），正常應 < 0.05
        "hold_elevator_std": float(np.nanstd(rec.column("elevator")[t < segs[0][1]])),
        "pitch_std_deg": float(np.nanstd(rec.column("pitch") - np.convolve(
            rec.column("pitch"), np.ones(50) / 50, mode="same"))),
    }


def plot(rec, segs, res, params, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = rec.column("t")
    t = t - t[0]
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(t, rec.column("u_cmd"), label="u_psi command")
    ax[0].plot(t, rec.column("psi_rate"), label="heading rate from body rates (P, Q, R)", alpha=0.4)
    ax[0].plot(t, rec.column("psi_rate_coord"), label="heading rate from roll: g·tanφ/V")
    for r in res["steps"]:
        if "tau_s" in r:
            ax[0].annotate(f"τ={r['tau_s']:.2f}s", (r["t_start"] + 1, r["cmd"]), fontsize=8)
    ax[0].set_ylabel("rad/s")
    ax[1].plot(t, rec.column("phi_cmd"), label="roll command")
    ax[1].plot(t, rec.column("roll"), label="roll")
    ax[1].set_ylabel("deg")
    ax[2].plot(t, rec.column("agl") - params["H"], label="AGL − H (m)")
    ax[2].plot(t, rec.column("elevator") * 10, label="elevator × 10", alpha=0.6)
    ax[2].set_ylabel("m")
    ax[3].plot(t, rec.column("airspeed") - params["V_t"], label="airspeed − V")
    ax[3].set_ylabel("m/s")
    ax[3].set_xlabel("t (s)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"Open-loop step test (τ median = {res['tau_median_s']})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-reset", action="store_true", help="不重新擺放飛機")
    ap.add_argument("--dry-run", action="store_true", help="用假的 X-Plane")
    ap.add_argument("--out", default=None, help="輸出資料夾（預設 results/xplane/<時間>_step_test）")
    add_profile_arg(ap)
    args = ap.parse_args(argv)

    link, clock, params, xp, k = setup(dry_run=args.dry_run, profile=args.profile)
    H, V = params["H"], params["V_t"]
    segs = schedule(xp["step_test"], params["u_max"])
    T = segs[-1][1]
    autopilot = BankToTurnAutopilot(xp["autopilot"], V_cmd=V, H_cmd=H)
    out = Path(args.out) if args.out else output_dir(xp, "step_test")
    out.mkdir(parents=True, exist_ok=True)

    st = link.read()
    if not args.no_reset:
        link.place(st.x, st.y, st.z - st.agl + H, st.psi, V, throttle=xp["autopilot"]["thr_trim"])
        clock.sleep(2.0)
        print(f"[擺放] 飛機離地 {link.read().agl:.0f} m（目標 {H:.0f} m）")
    print(f"[step test] 共 {T:.0f} s：先平飛 {segs[0][1]:.0f} s，再依序送 u_psi = {step_commands(xp['step_test'], params['u_max'])}")
    rec = Recorder()
    tick = 1.0 / xp["loop"]["control_hz"]
    min_agl = xp["loop"]["min_agl_ratio"] * H
    t0 = last = clock.now()
    link.pause(False)
    try:
        while True:
            now = clock.now()
            el = now - t0
            if el >= T:
                break
            dt, last = now - last, now
            u_cmd = next(c for a, b, c in segs if a <= el < b)
            st = link.read()
            if st.agl < min_agl:
                print(f"[step test] 離地 {st.agl:.0f} m 太低，結束")
                break
            elev, ail, rud, thr = autopilot.update(st, u_cmd, dt)
            link.controls(elev, ail, rud, thr)
            rec.add(t=now, u_cmd=u_cmd, psi_rate=st.psi_rate,
                    psi_rate_coord=-9.81 * math.tan(math.radians(st.roll_deg)) / max(st.airspeed, 1.0),
                    P=st.P, Q=st.Q, R=st.R, theta_dot=autopilot.last["theta_dot"],
                    roll=st.roll_deg, pitch=st.pitch_deg,
                    phi_cmd=autopilot.last["phi_cmd"], agl=st.agl, airspeed=st.airspeed,
                    elevator=elev, aileron=ail, rudder=rud, throttle=thr, x=st.x, y=st.y)
            clock.sleep(tick - (clock.now() - now))
    except KeyboardInterrupt:
        print("\n[step test] 使用者中斷")
    finally:
        link.pause(True)
        rec.save_csv(out / "log.csv")
        res = analyze(rec, segs, params) if len(rec.rows) > 10 else {"steps": [], "tau_median_s": None}
        save_json(out / "summary.json", {"result": res, "params": params, "xp": xp, "scale": k})
        if rec.rows:
            plot(rec, segs, res, params, out / "step_test.png")
        print(f"[step test] τ 中位數 = {res.get('tau_median_s')} s、延遲 = {res.get('delay_median_s')} s、"
              f"u_max 時穩態增益 = {res.get('gain_at_u_max')}")
        print(f"[step test] 平飛時升降舵抖動（標準差）= {res.get('hold_elevator_std')}（< 0.05 正常）、"
              f"俯仰高頻抖動 = {res.get('pitch_std_deg')}°")
        for r in res["steps"]:
            print(f"  u={r['cmd']:+.3f}  τ={r.get('tau_s', float('nan')):.2f}s  增益={r.get('gain', float('nan')):.2f}"
                  f"  坡度={r['bank_mean_last5s_deg']:+.1f}°  高度誤差最大={r['agl_max_err_m']:.1f} m"
                  f"  空速 {r['airspeed_min']:.1f}–{r['airspeed_max']:.1f}")
        print(f"[step test] 結果在 {out}")
    return out, res


if __name__ == "__main__":
    main()
