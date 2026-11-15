"""LSM 對照組（Yang et al. 2021, J. Intell. Robot. Syst. 102:81, Eq. 13–17）。"""
import numpy as np


def lsm_u_psi(u2, v2, V_t, alpha, f, params):
    """希望 ṡ2 = -λ·s2，用 Eq. 8 的結構以最小平方法解 u_psi。

    u̇2 = M1·u_psi + f/z_d·V_t、v̇2 = M2·u_psi
    u_psi = [M1(u̇2* - f/z_d·V_t) + M2·v̇2*] / (M1² + M2²)
    前作的深度用常數 z_d（HIL 為 100）。輸出限制在 ±u_max。
    """
    lam1, lam2 = params["lsm_lambda"]
    z_d = params["lsm_depth"]
    ca, sa = np.cos(alpha), np.sin(alpha)
    m1 = (f**2 + u2**2) / f * ca - v2 * sa
    m2 = u2 * v2 / f * ca + u2 * sa
    du_des, dv_des = -lam1 * u2, -lam2 * v2
    u_psi = (m1 * (du_des - f / z_d * V_t) + m2 * dv_des) / (m1**2 + m2**2)
    return float(np.clip(u_psi, -params["u_max"], params["u_max"]))
