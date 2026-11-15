"""Yang 2024 的 MPC：P1,soft（Eq. 30）、J2,soft（Eq. 38）、Jg,soft（Eq. 36）、切換（Eq. 39–43）。

決策變數 U = [u_psi(0|k) … u_psi(Np-1|k)]。
- |u_psi| ≤ u_max（Eq. 12）→ 變數上下限
- |Δu_psi| ≤ a_max（Eq. 13–14）→ 線性約束
- 感知範圍 g2（Eq. 27–28）、收縮約束 CC1 / CCg / CC2（Eq. 24, 35, 37）→ 成本裡的 β·max{·, 0}
預測模型 s2(i+1|k) = s2(i|k) + ṡ2·Δt（Eq. 8、10）。
"""
import time

import casadi as ca
import numpy as np

from . import geometry as geo
from .baselines import lsm_u_psi

MODES = ("J1", "J2", "Jg")


class YangMPC:
    def __init__(self, params):
        self.p = params
        self.alpha = geo.alpha_from(params["H"], params["R_d"])
        self.solvers = {m: self._build(m) for m in MODES}

    # ------------------------------------------------------------------ build
    def _depth(self, v2):
        p = self.p
        if p["depth_model"] == "const":
            return p["depth_const"]
        return geo.ideal_depth(v2, self.alpha, p["f"], p["H"])

    def _build(self, mode):
        p, a = self.p, self.alpha
        f, H, V, dt, Np = p["f"], p["H"], p["V_t"], p["dt"], p["N_p"]
        eps = p["smooth_eps"]
        q1, q2 = p["Q_s"]
        qg1, qg2 = p["Q_sg"]
        delta = p["delta"]

        U = ca.SX.sym("U", Np)
        par = ca.SX.sym("par", 3)  # u2(k), v2(k), u_psi(k-1)
        u, v = par[0], par[1]

        def norm(x, y):
            return ca.sqrt(x**2 + y**2 + eps)

        def absv(x):
            return ca.sqrt(x**2 + eps)

        hinge_eps = p["hinge_eps"]

        def hinge(x, scale):
            """max{x, 0}。hinge_eps = 0 時照論文用 max；> 0 時用平滑近似，
            ½(x + sqrt(x² + (eps·scale)²))，scale 是該項的典型大小。"""
            if hinge_eps <= 0:
                return ca.fmax(x, 0)
            return 0.5 * (x + ca.sqrt(x**2 + (hinge_eps * scale) ** 2))

        u3_0, v3_0 = geo.ideal_to_down(par[0], par[1], a, f)
        dU = [U[0] - par[2]] + [U[i] - U[i - 1] for i in range(1, Np)]

        J = 0
        first = None
        for i in range(Np):
            du, dv = geo.image_model(u, v, U[i], V, a, f, self._depth(v))
            u, v = u + du * dt, v + dv * dt  # s2(i+1|k)
            u3, v3 = geo.ideal_to_down(u, v, a, f)
            if i == 0:
                first = (u, v, u3, v3)
            effort = p["Q_u"] * dU[i] ** 2
            if mode == "J1":
                g2 = geo.perception_g(u3, v3, f, p["R_M"] - p["dR"], H)
                J += q1 * u**2 + q2 * v**2 + effort + p["beta_1"] * hinge(g2, 1.0)
            else:  # J2 與 Jg 的逐步成本都是 l_g（Eq. 33、38）
                J += qg1 * u3**2 + qg2 * v3**2 + effort

        u1s, v1s, u31, v31 = first
        if mode == "J1":  # Eq. 24 → 29
            cc1 = norm(u1s, v1s) - delta * norm(par[0], par[1])
            J += p["beta_2"] * hinge(cc1, f)
        elif mode == "Jg":  # Eq. 35 → 36
            ccg = norm(u31, v31) - delta * norm(u3_0, v3_0)
            J += p["beta_g"] * hinge(ccg, f)
        else:  # Eq. 37 → 38
            if p["cc2_on"] == "s3":
                (a1, b1), (a0, b0) = (u31, v31), (u3_0, v3_0)
            else:
                (a1, b1), (a0, b0) = (u1s, v1s), (par[0], par[1])
            cc2a = absv(a1) - delta * absv(a0)
            cc2b = absv(b1) - delta * absv(b0)
            J += p["a_2"] * (hinge(cc2a, f) + hinge(cc2b, f))

        nlp = {"x": U, "p": par, "f": J, "g": ca.vertcat(*dU)}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": p["ipopt_max_iter"],
            "ipopt.tol": p["ipopt_tol"],
            "print_time": False,
        }
        return ca.nlpsol(f"yang_{mode}", "ipopt", nlp, opts)

    # ------------------------------------------------------------------ solve
    def solve(self, mode, u2, v2, u_prev, U0):
        p = self.p
        Np = p["N_p"]
        lb = -p["u_max"] * np.ones(Np)
        t0 = time.perf_counter()
        sol = self.solvers[mode](
            x0=U0, p=[u2, v2, u_prev],
            lbx=lb, ubx=-lb,
            lbg=-p["a_max"] * np.ones(Np), ubg=p["a_max"] * np.ones(Np),
        )
        elapsed = time.perf_counter() - t0
        stats = self.solvers[mode].stats()
        U = np.array(sol["x"]).ravel()
        return U, {"time": elapsed, "status": stats["return_status"],
                   "success": stats["success"], "iters": stats["iter_count"],
                   "cost": float(sol["f"])}

    # ------------------------------------------------------------ warm start
    def lsm_warm_start(self, u2, v2, u_prev):
        """Eq. 31：k = 0 時用 LSM 加預測模型推出初始序列，再修成滿足 Eq. 12–13。"""
        p, a = self.p, self.alpha
        U = np.zeros(p["N_p"])
        last = u_prev
        for i in range(p["N_p"]):
            u_cmd = lsm_u_psi(u2, v2, p["V_t"], a, p["f"], p)
            u_cmd = np.clip(u_cmd, last - p["a_max"], last + p["a_max"])
            U[i] = last = u_cmd
            du, dv = geo.image_model(u2, v2, u_cmd, p["V_t"], a, p["f"], self._depth(v2))
            u2, v2 = u2 + du * p["dt"], v2 + dv * p["dt"]
        return U

    @staticmethod
    def shift(U):
        """Eq. 32：上一步的最佳解往前平移一格，最後一格重複。"""
        return np.concatenate([U[1:], U[-1:]])


class Switcher:
    """計數器 n1（Eq. 41–43）與模式選擇（Eq. 40）。"""

    MAPPINGS = {
        "text": ("Jg", "J2", "J1"),
        "proof": ("Jg", "J1", "J2"),
        "literal": ("J1", "J2", "Jg"),
    }

    def __init__(self, params, g1_initial):
        self.n_max = params["n_max"]
        self.mapping = self.MAPPINGS[params["switch_mapping"]]
        self.n1 = self.n_max if g1_initial < 0 else 0  # Eq. 43

    def update(self, g1):
        self.n1 = min(self.n1 + 1, self.n_max) if g1 <= 0 else 0  # Eq. 41–42

    @property
    def mode(self):
        zero, middle, full = self.mapping
        if self.n1 == 0:
            return zero
        return full if self.n1 >= self.n_max else middle
