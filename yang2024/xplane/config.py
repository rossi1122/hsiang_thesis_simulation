"""讀 X-Plane 設定，並把論文參數換算到 SP80 的尺度。

縮放規則（保持論文的無因次量不變）：
  k_L = H / H_paper、k_V = V_t / V_paper、k_T = k_L / k_V、k_f = f_xp / f_paper
  長度 ×k_L：H、R_d、R_M、ΔR、lsm_depth、depth_const
  時間 ×k_T：Δt（N_p、n_max 步數不變 → 預測時域、切換門檻的「秒數」也 ×k_T）
  角速度 ÷k_T：u_max、u_psi_init、lsm_lambda
  Δu_psi 每步上限：角加速度 ÷k_T²，再乘新的 Δt → a_max ÷k_T
  影像：像素 ×k_f；Q_s、Q_sg ÷k_f²（讓 q·s² 不變），Q_u ×k_T²（讓 Q_u·Δu² 不變）
  其他（δ、β、n_max、雲台 λ 的「每步角度」）本來就是無因次，不變。
"""
from pathlib import Path

import numpy as np
import yaml

from .. import REPO_DIR, load_params

XP_DIR = Path(__file__).resolve().parent


def camera_f(cam):
    """f = (W/2) / tan(HFOV/2)。"""
    return cam["render_width"] / 2.0 / np.tan(np.radians(cam["hfov_deg"]) / 2.0)


def load_xp(path=None):
    with open(path or XP_DIR / "xp_params.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def scale_params(paper, V_t, H, f):
    """論文參數（dict）→ X-Plane 尺度。回傳新的 dict 和縮放因子。"""
    kL = H / paper["H"]
    kV = V_t / paper["V_t"]
    kT = kL / kV
    kf = f / paper["f"]
    p = dict(paper)
    p.update(
        V_t=V_t, H=H, f=f,
        R_d=paper["R_d"] * kL, R_M=paper["R_M"] * kL, dR=paper["dR"] * kL,
        lsm_depth=paper["lsm_depth"] * kL, depth_const=paper["depth_const"] * kL,
        dt=paper["dt"] * kT,
        u_max=paper["u_max"] / kT, u_psi_init=paper["u_psi_init"] / kT,
        a_max=paper["a_max"] / kT,
        lsm_lambda=[lam / kT for lam in paper["lsm_lambda"]],
        Q_s=[q / kf**2 for q in paper["Q_s"]], Q_sg=[q / kf**2 for q in paper["Q_sg"]],
        Q_u=paper["Q_u"] * kT**2,
        uav_init=[c * kL for c in paper["uav_init"]],
    )
    return p, {"k_L": kL, "k_V": kV, "k_T": kT, "k_f": kf}


PROFILES = ("scaled", "original")


def _merge(base, over):
    for key, val in (over or {}).items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _merge(base[key], val)
        else:
            base[key] = val
    return base


def build(xp_path=None, profile="scaled", **overrides):
    """回傳 (MPC 用的參數, X-Plane 設定, 縮放因子)。

    profile：xp_params.yaml 裡 profiles 的名字（scaled = 尺度版、original = 論文原值），覆寫共用設定。
    overrides 再覆寫 scaling 的 V_t、H。
    """
    xp = load_xp(xp_path)
    profiles = xp.pop("profiles", {}) or {}
    if profile not in profiles:
        raise ValueError(f"xp_params.yaml 沒有 profile「{profile}」，可用：{list(profiles)}")
    _merge(xp, profiles[profile])
    xp["profile"] = profile
    xp["scaling"].update({k: v for k, v in overrides.items() if k in ("V_t", "H")})
    f = camera_f(xp["camera"])
    params, k = scale_params(load_params(), xp["scaling"]["V_t"], xp["scaling"]["H"], f)
    # 實際雲台角度範圍、影像尺寸改成 X-Plane 的
    params["image_size"] = [xp["camera"]["render_width"], xp["camera"]["capture_height"]]
    return params, xp, k


def log_dir(xp):
    d = REPO_DIR / xp["loop"]["log_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d
