"""傾斜轉彎內迴路：u_psi → 滾轉角 → 副翼；方向舵協調轉彎；定高、定速。

Yang 2024 只輸出 u_psi（航向變化率），沒說怎麼轉成飛控命令；本專案採協調轉彎：
  φ_cmd = atan(V·u_psi / g)（CLAUDE.md 4.5、thesis_context 3.1）
學長的內迴路是「方向舵轉向、副翼保持水平」，固定翼大角度轉彎做不到 Yang 要的 0.5 rad/s 等級，所以改寫。
角度用 deg（沿用學長的增益量級）；本專案 u_psi 逆時針為正，X-Plane 滾轉右翼下沉為正，
所以左轉（u_psi > 0）需要負的滾轉角。
"""
import math

import numpy as np

G = 9.81


class BankToTurnAutopilot:
    def __init__(self, gains, V_cmd, H_cmd):
        self.g = gains
        self.V_cmd = V_cmd
        self.H_cmd = H_cmd
        self.i_phi = 0.0
        self.i_h = 0.0
        self.i_v = 0.0
        self.last = {}

    def phi_command_deg(self, u_psi, V):
        phi = -math.degrees(math.atan(max(V, 1.0) * u_psi / G))
        lim = self.g["bank_limit_deg"]
        return float(np.clip(phi, -lim, lim))

    def update(self, st, u_psi, dt):
        g = self.g
        dt = max(dt, 1e-3)
        # ---------------- 橫向：滾轉 → 副翼
        phi_cmd = self.phi_command_deg(u_psi, st.airspeed)
        e_phi = phi_cmd - st.roll_deg
        self.i_phi = float(np.clip(self.i_phi + e_phi * dt, -50.0, 50.0))
        aileron = g["k_phi"] * e_phi + g["k_phi_i"] * self.i_phi - g["k_p"] * st.P
        aileron = float(np.clip(aileron, -g["ail_limit"], g["ail_limit"]))
        # 協調轉彎：機體偏航角速度 r_ref = g·sinφ / V（deg/s），方向舵正 = 機頭往右
        r_ref = math.degrees(G * math.sin(math.radians(st.roll_deg)) / max(st.airspeed, 1.0))
        rudder = float(np.clip(g["k_r"] * (r_ref - st.R), -g["rud_limit"], g["rud_limit"]))
        # ---------------- 縱向：高度 → 俯仰角 → 俯仰率 → 升降舵
        e_h = self.H_cmd - st.agl
        self.i_h = float(np.clip(self.i_h + e_h * dt, -500.0, 500.0))
        bank = math.radians(min(abs(st.roll_deg), 60.0))
        theta_cmd = (g["theta_trim_deg"] + g["k_h"] * e_h + g["k_h_i"] * self.i_h - g["k_hdot"] * st.vz
                     + g["bank_pitch_comp_deg"] * (1.0 / math.cos(bank) - 1.0))
        theta_cmd = float(np.clip(theta_cmd, *g["theta_limit_deg"]))
        q_cmd = g["K_theta"] * (theta_cmd - st.pitch_deg)
        elevator = float(np.clip(g["K_q"] * (q_cmd - st.Q), -g["elev_limit"], g["elev_limit"]))
        # ---------------- 空速 → 油門
        e_v = self.V_cmd - st.airspeed
        self.i_v = float(np.clip(self.i_v + e_v * dt, -20.0, 20.0))
        throttle = float(np.clip(g["thr_trim"] + g["k_v"] * e_v + g["k_v_i"] * self.i_v, *g["thr_limit"]))

        self.last = dict(phi_cmd=phi_cmd, theta_cmd=theta_cmd, r_ref=r_ref)
        return elevator, aileron, rudder, throttle
