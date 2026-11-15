"""Phase 1 第 3 步：影像運動拆解（回答 RQ1：Yang 2024 的影像誤差有多少來自沒建模的運動？）

畫面上目標的移動來自四個來源（CLAUDE.md 1.1）：
  ① 飛機前進 + 偏航（V_t、ψ̇）   ② 飛機姿態動態（偏航率延遲、滾轉）
  ③ 雲台轉動                       ④ 目標移動

A. Ideal State 特徵 s2（Yang 的 MPC 預測的就是它）
   每一步：實際 ṡ2（有限差分）− Eq. 8 預測（用命令 u_psi）= 預測誤差 e，再用影像雅可比拆成
     ④ 目標：L_t(s2, z)·(−v_T 在 Ideal 相機座標)
     ② 偏航延遲：B(s2)·(實際 ψ̇ − 命令 u_psi)，B 是 Eq. 8 中 u_psi 那一欄
     殘差：離散化等其他誤差
   ③ 雲台不會出現在 s2：Ideal State 轉換本來就把雲台角度轉掉了（只要雲台角度已知且同步）。
B. N_p 步（0.75 s）開迴路預測誤差：Yang 原版 vs 加入目標運動 vs 加入偏航延遲 vs 兩者都加
   （都用實際施加的輸入，評估的是「預測模型」本身）。
C. 真實相機特徵 s1 的運動預算：①②③④ 各自造成多少 px/s（雲台在這裡才有作用）。
D. 驗證：遠距離時 J1（P1,soft）為什麼先往左轉、追丟。

用法（repo 根目錄）：python -m yang2024.motion_budget
"""
import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import REPO_DIR, load_params  # noqa: E402
from . import geometry as geo  # noqa: E402
from .mpc import YangMPC  # noqa: E402
from .sim import simulate  # noqa: E402

# 名稱, 說明, 參數覆寫, simulate 參數
SCENARIOS = [
    ("static", "靜止目標（驗證拆解方法，誤差應接近 0）",
     {}, dict(Vc=0.0, T=60.0)),
    ("fig4", "論文 Fig. 4 條件：V_c = 7、近距離起始",
     {}, dict(Vc=7.0, T=150.0)),
    ("fig4_lag", "同 Fig. 4，加偏航延遲 τ = 0.5 s 與滾轉",
     {"yaw_tau": 0.5, "bank_in_camera": True}, dict(Vc=7.0, T=150.0)),
    ("fig5", "論文 Fig. 5 條件：V_c = 9",
     {}, dict(Vc=9.0, T=150.0)),
    ("far_v15", "後方 300 m 起始、V_c = 15（本論文情境）",
     {}, dict(Vc=15.0, T=350.0, uav_init=(-300.0, 0.0), uav_heading=0.0)),
    ("far_v15_lag", "同上，加偏航延遲 τ = 0.5 s 與滾轉",
     {"yaw_tau": 0.5, "bank_in_camera": True},
     dict(Vc=15.0, T=350.0, uav_init=(-300.0, 0.0), uav_heading=0.0)),
]


# --------------------------------------------------------------------- 工具
def jacobian_full(u, v, z, f):
    """Eq. 6 的影像雅可比（2×6）。"""
    return np.array([
        [-f / z, 0.0, u / z, u * v / f, -(f**2 + u**2) / f, v],
        [0.0, -f / z, v / z, (f**2 + v**2) / f, -u * v / f, -u],
    ])


def yaw_column(u, v, alpha, f):
    """Eq. 8 中 u_psi 那一欄 B(s2)。"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([(f**2 + u**2) / f * ca - v * sa, u * v / f * ca + u * sa])


C_IDEAL_CACHE = {}


def target_term_ideal(u, v, psi, tvx, tvy, alpha, f, H):
    """④ 目標移動對 ṡ2 的貢獻：等同相機以 -v_T 移動。"""
    key = round(alpha, 12)
    if key not in C_IDEAL_CACHE:
        C_IDEAL_CACHE[key] = geo.camera_rows(-np.pi / 2, -alpha)
    v_cam = C_IDEAL_CACHE[key] @ (geo.rot_z(psi).T @ np.array([tvx, tvy, 0.0]))
    z = geo.ideal_depth(v, alpha, f, H)
    return jacobian_full(u, v, z, f)[:, :3] @ (-v_cam)


def rms(x):
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(x**2, axis=-1)))) if x.ndim > 1 else float(np.sqrt(np.mean(x**2)))


# ------------------------------------------------------------- A. s2 拆解
def s2_decomposition(log, p):
    f, V, dt, H = p["f"], p["V_t"], p["dt"], p["H"]
    alpha = geo.alpha_from(H, p["R_d"])
    s2 = np.column_stack([log["u2"], log["v2"]])
    ok = log["ideal_valid"].astype(bool)
    idx = np.where(ok[:-1] & ok[1:])[0]
    actual = (s2[idx + 1] - s2[idx]) / dt
    pred, tgt, lag = [], [], []
    for k in idx:
        u, v = s2[k]
        z = geo.ideal_depth(v, alpha, f, H)
        pred.append(geo.image_model(u, v, log["u_psi"][k], V, alpha, f, z))
        tgt.append(target_term_ideal(u, v, log["psi"][k], log["tvx"][k], log["tvy"][k], alpha, f, H))
        lag.append(yaw_column(u, v, alpha, f) * (log["psi_rate"][k] - log["u_psi"][k]))
    pred, tgt, lag = np.array(pred), np.array(tgt), np.array(lag)
    err = actual - pred
    resid = err - tgt - lag
    e2 = np.mean(np.sum(err**2, axis=1))

    def explained(remain):
        return float(1.0 - np.mean(np.sum(remain**2, axis=1)) / e2) if e2 > 0 else float("nan")

    return {
        "t": log["t"][idx], "actual": actual, "pred": pred, "err": err,
        "target": tgt, "lag": lag, "resid": resid,
        "summary": {
            "actual_rms_px_s": rms(actual), "pred_rms_px_s": rms(pred), "err_rms_px_s": rms(err),
            "err_over_actual": rms(err) / rms(actual) if rms(actual) > 0 else float("nan"),
            "target_rms_px_s": rms(tgt), "lag_rms_px_s": rms(lag), "resid_rms_px_s": rms(resid),
            "explained_by_target": explained(err - tgt),
            "explained_by_lag": explained(err - lag),
            "explained_by_target_and_lag": explained(resid),
            "valid_ratio": float(ok.mean()),
        },
    }


# ---------------------------------------------------- B. N_p 步預測誤差
def horizon_errors(log, p, stride=2):
    f, V, dt, H, Np = p["f"], p["V_t"], p["dt"], p["H"], p["N_p"]
    alpha = geo.alpha_from(H, p["R_d"])
    n = len(log["t"])
    ok = log["ideal_valid"].astype(bool)
    variants = {"yang": (False, False), "+target": (True, False),
                "+lag": (False, True), "+target+lag": (True, True)}
    errs = {k: [] for k in variants}
    for k in range(0, n - Np, stride):
        if not ok[k:k + Np + 1].all():
            continue
        truth = np.array([log["u2"][k + Np], log["v2"][k + Np]])
        for name, (use_tgt, use_lag) in variants.items():
            u, v, psi = log["u2"][k], log["v2"][k], log["psi"][k]
            for i in range(Np):
                rate = log["psi_rate"][k + i] if use_lag else log["u_psi"][k + i]
                z = geo.ideal_depth(v, alpha, f, H)
                du, dv = geo.image_model(u, v, rate, V, alpha, f, z)
                if use_tgt:
                    tu, tv = target_term_ideal(u, v, psi, log["tvx"][k + i], log["tvy"][k + i], alpha, f, H)
                    du, dv = du + tu, dv + tv
                u, v, psi = u + du * dt, v + dv * dt, psi + rate * dt
            errs[name].append(np.hypot(u - truth[0], v - truth[1]))
    return {name: rms(np.array(e)) for name, e in errs.items()}


# --------------------------------------------------- C. s1 運動預算
def s1_budget(log, p):
    f, V, dt = p["f"], p["V_t"], p["dt"]
    n = len(log["t"]) - 1
    d_pan = np.angle(np.exp(1j * np.diff(log["pan"]))) / dt
    d_tilt = np.diff(log["tilt"]) / dt
    d_roll = np.diff(log["roll"]) / dt
    s1 = np.column_stack([log["u1"], log["v1"]])
    actual = np.diff(s1, axis=0) / dt
    parts = {"①飛機前進+偏航": [], "②滾轉": [], "③雲台": [], "④目標": []}
    for k in range(n):
        L = jacobian_full(log["u1"][k], log["v1"][k], log["depth1"][k], f)
        Rwb = geo.body_to_world(log["psi"][k], 0.0, log["roll"][k])
        C = geo.camera_rows(log["pan"][k], log["tilt"][k])
        to_cam = C @ Rwb.T
        v_air = to_cam @ np.array([V * np.cos(log["psi"][k]), V * np.sin(log["psi"][k]), 0.0])
        w_yaw = to_cam @ np.array([0.0, 0.0, log["psi_rate"][k]])
        parts["①飛機前進+偏航"].append(L @ np.concatenate([v_air, w_yaw]))
        parts["②滾轉"].append(L[:, 3:] @ (C @ np.array([d_roll[k], 0.0, 0.0])))
        w_gim = C @ np.array([0.0, 0.0, d_pan[k]]) + np.array([d_tilt[k], 0.0, 0.0])
        parts["③雲台"].append(L[:, 3:] @ w_gim)
        v_tgt = to_cam @ np.array([log["tvx"][k], log["tvy"][k], 0.0])
        parts["④目標"].append(L[:, :3] @ (-v_tgt))
    parts = {k: np.array(v) for k, v in parts.items()}
    total = sum(parts.values())
    out = {k: rms(v) for k, v in parts.items()}
    out["實際 ṡ1"] = rms(actual)
    out["殘差（實際 − 四項和）"] = rms(actual - total)
    out["①+②+④（不含雲台）"] = rms(parts["①飛機前進+偏航"] + parts["②滾轉"] + parts["④目標"])
    return out


# ------------------------------------------------ D. J1 遠距離行為
def j1_far_check(p, out_dir):
    """P1,soft 從後方 300 m、V_c = 15：看目標相對方位，以及加長預測時域是否改善。"""
    res = {}
    logs = {}
    for Np in (15, 60):
        pp = dict(p, N_p=Np)
        mpc = YangMPC(pp)
        t0 = time.time()
        lg = simulate(pp, "p1soft", Vc=15.0, T=150.0, uav_init=(-300.0, 0.0), uav_heading=0.0, mpc=mpc)
        rel = np.column_stack([lg["tx"] - lg["x"], lg["ty"] - lg["y"]])
        bearing = np.degrees(np.angle(np.exp(1j * (np.arctan2(rel[:, 1], rel[:, 0]) - lg["psi"]))))
        logs[Np] = (lg, bearing)
        res[f"N_p={Np}"] = {
            "R_at_150s_m": float(lg["R"][-1]),
            "bearing_at_20s_deg": float(bearing[int(20 / p["dt"])]),
            "heading_change_at_20s_deg": float(np.degrees(lg["psi"][int(20 / p["dt"])] - lg["psi"][0])),
            "wall_time_s": round(time.time() - t0, 1),
        }
    # t = 0 時 Eq. 8 兩欄的大小：轉彎 vs 前進 對 u̇2 的影響
    lg = logs[15][0]
    alpha = geo.alpha_from(p["H"], p["R_d"])
    u, v = lg["u2"][0], lg["v2"][0]
    z = geo.ideal_depth(v, alpha, p["f"], p["H"])
    Bu = yaw_column(u, v, alpha, p["f"])[0]
    res["t0"] = {"u2_px": float(u), "v2_px": float(v),
                 "du2_by_turning_umax_px_s": float(abs(Bu) * p["u_max"]),
                 "du2_by_flying_forward_px_s": float(p["f"] / z * p["V_t"])}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for Np, (lg, bearing) in logs.items():
        ax[0].plot(lg["t"], bearing, label=f"N_p = {Np}")
        ax[1].plot(lg["t"], lg["R"], label=f"N_p = {Np}")
    ax[0].axhline(-90, color="k", ls="--", lw=1)
    ax[0].set(xlabel="t (s)", ylabel="target bearing (deg, -90 = right side)",
              title="P1,soft, start 300 m behind, V_c = 15")
    ax[1].axhline(p["R_M"], color="k", ls="--", lw=1)
    ax[1].set(xlabel="t (s)", ylabel="R (m)", title="horizontal distance")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "j1_far_check.png", dpi=120)
    plt.close(fig)
    return res


# ------------------------------------------------------------------ 圖
def plot_timeseries(dec, title, path, t_max=None):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t = dec["t"]
    m = slice(None) if t_max is None else t <= t_max
    for i, comp in enumerate(("u̇2", "v̇2")):
        ax[i].plot(t[m], dec["err"][m, i], color="k", lw=1.4, label="prediction error (actual − Eq. 8)")
        ax[i].plot(t[m], dec["target"][m, i], color="#e76f51", lw=1.0, label="④ target motion")
        ax[i].plot(t[m], dec["lag"][m, i], color="#2a9d8f", lw=1.0, label="② yaw-rate lag")
        ax[i].plot(t[m], dec["resid"][m, i], color="#999999", lw=0.8, label="residual")
        ax[i].set_ylabel(f"{comp} error (px/s)")
        ax[i].grid(alpha=0.3)
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].set_xlabel("t (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bars(table, keys, labels, ylabel, title, path):
    names = list(table)
    x = np.arange(len(names))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (k, lab) in enumerate(zip(keys, labels)):
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, [table[n][k] for n in names], w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ 主程式
def main():
    base = load_params()
    out_dir = REPO_DIR / "results" / f"{datetime.now():%Y%m%d}_yang2024_motion_budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    mpc_cache = {}
    A, B, C = {}, {}, {}
    decs = {}
    for name, desc, over, kw in SCENARIOS:
        p = dict(base, **over)
        key = tuple(sorted(over.items()))
        if key not in mpc_cache:
            mpc_cache[key] = YangMPC(p)
        t0 = time.time()
        log = simulate(p, "switch", mpc=mpc_cache[key], **kw)
        dec = s2_decomposition(log, p)
        decs[name] = dec
        A[name] = dec["summary"]
        B[name] = horizon_errors(log, p)
        C[name] = s1_budget(log, p)
        print(f"{name:12s} ({time.time() - t0:5.1f} s)  e_rms={A[name]['err_rms_px_s']:8.1f} px/s  "
              f"④ 解釋 {A[name]['explained_by_target']:.2f}  ④+② 解釋 {A[name]['explained_by_target_and_lag']:.2f}  "
              f"N_p 誤差 Yang {B[name]['yang']:.1f} → +④ {B[name]['+target']:.1f} → +④② {B[name]['+target+lag']:.1f} px")

    D = j1_far_check(base, out_dir)
    print("J1 遠距離檢查：", json.dumps(D, ensure_ascii=False))

    plot_timeseries(decs["fig4_lag"], "s2 prediction error decomposition: Fig. 4 setting + yaw lag (first 60 s)",
                    out_dir / "s2_error_fig4_lag.png", t_max=60)
    plot_timeseries(decs["far_v15"], "s2 prediction error decomposition: 300 m behind, V_c = 15",
                    out_dir / "s2_error_far_v15.png")
    plot_bars(A, ["err_rms_px_s", "target_rms_px_s", "lag_rms_px_s", "resid_rms_px_s"],
              ["prediction error", "④ target", "② yaw lag", "residual"], "RMS (px/s)",
              "Eq. 8 one-step prediction error and its sources (s2)", out_dir / "s2_error_bars.png")
    plot_bars(B, ["yang", "+target", "+lag", "+target+lag"],
              ["Yang Eq. 8", "+ ④ target", "+ ② yaw lag", "+ ④ + ②"], "RMS error at N_p (px)",
              "0.75 s (N_p = 15) open-loop prediction error of s2", out_dir / "horizon_error_bars.png")
    s1_keys = ["實際 ṡ1", "①飛機前進+偏航", "②滾轉", "③雲台", "④目標", "①+②+④（不含雲台）"]
    plot_bars(C, s1_keys, ["actual ds1/dt", "① aircraft", "② roll", "③ gimbal", "④ target",
                           "①+②+④ (no gimbal)"], "RMS (px/s)",
              "Real-camera image motion budget (s1)", out_dir / "s1_budget_bars.png")

    def table_md():
        lines = ["## A. s2 一步預測誤差（Eq. 8）", "",
                 "| 情境 | 實際 ṡ2 RMS | Eq. 8 誤差 RMS | 誤差/實際 | ④ 目標 | ② 延遲 | 殘差 | ④ 可解釋 | ④+② 可解釋 |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for n, d in A.items():
            lines.append(f"| {n} | {d['actual_rms_px_s']:.1f} | {d['err_rms_px_s']:.1f} | {d['err_over_actual']:.2f} "
                         f"| {d['target_rms_px_s']:.1f} | {d['lag_rms_px_s']:.1f} | {d['resid_rms_px_s']:.1f} "
                         f"| {d['explained_by_target']:.2f} | {d['explained_by_target_and_lag']:.2f} |")
        lines += ["", "（單位 px/s；「可解釋」= 1 − 剩餘誤差能量 / 原誤差能量）", "",
                  "## B. 0.75 s（N_p = 15）開迴路預測誤差（px）", "",
                  "| 情境 | Yang Eq. 8 | + ④ 目標 | + ② 延遲 | + ④ + ② |", "|---|---|---|---|---|"]
        for n, d in B.items():
            lines.append(f"| {n} | {d['yang']:.1f} | {d['+target']:.1f} | {d['+lag']:.1f} | {d['+target+lag']:.1f} |")
        lines += ["", "## C. 真實相機 s1 運動預算（RMS, px/s）", "",
                  "| 情境 | " + " | ".join(s1_keys) + " | 殘差 |", "|---" * (len(s1_keys) + 2) + "|"]
        for n, d in C.items():
            lines.append(f"| {n} | " + " | ".join(f"{d[k]:.1f}" for k in s1_keys)
                         + f" | {d['殘差（實際 − 四項和）']:.1f} |")
        lines += ["", "## D. J1（P1,soft）遠距離檢查（後方 300 m、V_c = 15）", "", "```",
                  json.dumps(D, ensure_ascii=False, indent=2), "```"]
        return "\n".join(lines) + "\n"

    (out_dir / "summary.md").write_text(table_md(), encoding="utf-8")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"date": datetime.now().isoformat(timespec="seconds"),
                   "scenarios": [(n, d, o) for n, d, o, _ in SCENARIOS],
                   "s2_one_step": A, "horizon": B, "s1_budget": C, "j1_far": D},
                  fh, ensure_ascii=False, indent=2)
    print(f"結果存在 {out_dir.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
