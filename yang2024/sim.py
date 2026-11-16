"""數值閉迴路模擬（Section 4）：unicycle 飛機 + 雲台 + 針孔相機 + 地面目標。

每一步（Algorithm 1）：
  1. 投影得到真實相機特徵 (u1, v1)
  2. 轉成 Ideal State (u2, v2)（Eq. 7）、垂直相機 (u3, v3)（Eq. 54）、g1（Eq. 20）
  3. 控制器算 u_psi：LSM / P1,soft / 切換式 MPC
  4. 雲台律（Eq. 9）更新 pan / tilt
  5. 飛機（Eq. 1）與目標前進一步

可選的「真實世界」效果（params.yaml，預設關閉 = 論文的數值模擬條件）：
  yaw_tau        > 0：實際偏航率以一階延遲追隨 u_psi 命令（論文假設 u_psi 立即生效）
  bank_in_camera true：協調轉彎的滾轉角 φ = atan(V·ψ̇/g) 會帶動相機（量測端轉 Ideal State 時已知 φ）
"""
import numpy as np

from . import geometry as geo
from .baselines import lsm_u_psi
from .models import Gimbal, Target, unicycle_step
from .mpc import Switcher, YangMPC

CONTROLLERS = ("lsm", "p1soft", "switch")

LOG_KEYS = ("t", "x", "y", "psi", "tx", "ty", "u_psi", "mode", "n1",
            "u1", "v1", "u2", "v2", "u3", "v3", "g1", "R", "pan", "tilt",
            "solve_time", "solve_ok", "visible", "in_fov", "ideal_valid", "target_speed",
            "psi_rate", "roll", "depth1", "pan_rate", "tilt_rate", "tvx", "tvy")


def simulate(params, controller="switch", Vc=7.0, T=150.0, target_profile=None,
             uav_init=None, uav_heading=None, target_init=None, target_heading=None,
             mpc=None, verbose=False):
    """跑一次閉迴路。回傳 dict（每個 key 是一條時間序列的 numpy 陣列）。"""
    if controller not in CONTROLLERS:
        raise ValueError(f"controller 必須是 {CONTROLLERS}")
    p = params
    f, H, V, dt = p["f"], p["H"], p["V_t"], p["dt"]
    alpha = geo.alpha_from(H, p["R_d"])
    W, Himg = p["image_size"]

    x, y = uav_init if uav_init is not None else p["uav_init"]
    psi = p["uav_heading_init"] if uav_heading is None else uav_heading
    tx, ty = target_init if target_init is not None else p["target_init"]
    th = p["target_heading_init"] if target_heading is None else target_heading
    target = Target(tx, ty, th, speed=Vc, profile=target_profile)

    uav_pos = np.array([x, y, H])
    pan, tilt = geo.pointing_angles(target.position, uav_pos, psi)
    gimbal = Gimbal(pan, tilt, p)

    if controller != "lsm" and mpc is None:
        mpc = YangMPC(p)

    log = {k: [] for k in LOG_KEYS}
    u_prev = p["u_psi_init"]
    U = None
    switcher = None
    last_s1 = (0.0, 0.0)
    last_s2 = (0.0, 0.0)
    steps = int(round(T / dt))
    tau = p.get("yaw_tau", 0.0)
    bank = p.get("bank_in_camera", False)
    g_acc = p.get("gravity", 9.81)
    psi_rate = p["u_psi_init"]  # 實際偏航率

    for k in range(steps):
        t = k * dt
        uav_pos = np.array([x, y, H])
        # 右轉（ψ̇ < 0）時右翼下沉，roll 為正
        roll = float(-np.arctan(V * psi_rate / g_acc)) if bank else 0.0
        u1, v1, depth = geo.project(target.position, uav_pos, psi, gimbal.pan, gimbal.tilt, f, roll=roll)
        visible = depth > 0
        if not visible:  # 目標在相機後方：沿用上一次的量測
            u1, v1 = last_s1
        last_s1 = (u1, v1)
        in_fov = visible and abs(u1) <= W / 2 and abs(v1) <= Himg / 2

        u2, v2, u3, v3, ideal_valid = geo.ideal_and_down(u1, v1, gimbal.pan, gimbal.tilt, alpha, f,
                                                         roll=roll)
        if ideal_valid:
            last_s2 = (u2, v2)
        else:  # 目標在 Ideal 相機後方（論文未定義）：控制器沿用上一次有效的 s2
            u2, v2 = last_s2
        g1 = geo.perception_g(u3, v3, f, p["R_M"], H)

        mode, n1, solve_time, ok = "LSM", -1, 0.0, True
        if controller == "lsm":
            u_psi = lsm_u_psi(u2, v2, V, alpha, f, p)
        else:
            if controller == "p1soft":
                mode = "J1"
            else:
                if switcher is None:
                    switcher = Switcher(p, g1)
                else:
                    switcher.update(g1)
                mode, n1 = switcher.mode, switcher.n1
            U0 = mpc.lsm_warm_start(u2, v2, u_prev) if U is None else YangMPC.shift(U)
            U, info = mpc.solve(mode, u2, v2, u_prev, U0)
            u_psi, solve_time, ok = float(U[0]), info["time"], info["success"]
            if verbose and not ok:
                print(f"[t={t:6.2f}] {mode} 求解失敗：{info['status']}")

        # 這一步實際的偏航率（τ = 0 時就是命令）
        if tau > 0:
            psi_rate += (1.0 - np.exp(-dt / tau)) * (u_psi - psi_rate)
        else:
            psi_rate = u_psi
        pan_rate, tilt_rate = gimbal.rates(u1, v1)
        t_speed, _ = target.command(t)

        rec = dict(t=t, x=x, y=y, psi=psi, tx=target.x, ty=target.y, u_psi=u_psi,
                   mode=mode, n1=n1, u1=u1, v1=v1, u2=u2, v2=v2, u3=u3, v3=v3, g1=g1,
                   R=float(np.hypot(target.x - x, target.y - y)),
                   pan=gimbal.pan, tilt=gimbal.tilt, solve_time=solve_time,
                   solve_ok=ok, visible=visible, in_fov=in_fov, ideal_valid=ideal_valid,
                   target_speed=t_speed, psi_rate=psi_rate, roll=roll, depth1=depth,
                   pan_rate=pan_rate, tilt_rate=tilt_rate,
                   tvx=t_speed * np.cos(target.heading), tvy=t_speed * np.sin(target.heading))
        for key in LOG_KEYS:
            log[key].append(rec[key])

        gimbal.step(pan_rate, tilt_rate)
        x, y, psi = unicycle_step(x, y, psi, V, psi_rate, dt)
        target.step(t, dt)
        u_prev = u_psi

    return {k: np.asarray(v) for k, v in log.items()}
