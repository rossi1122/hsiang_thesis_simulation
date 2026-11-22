"""飛機（Eq. 1）、雲台（Eq. 2 + 前作 Eq. 19–20 / 2024 Eq. 9）、目標運動。"""
import numpy as np


def unicycle_step(x, y, psi, V, u_psi, dt):
    """Eq. 1 的精確離散化（u_psi 在 dt 內固定）。"""
    if abs(u_psi) < 1e-9:
        return x + V * np.cos(psi) * dt, y + V * np.sin(psi) * dt, psi
    psi_next = psi + u_psi * dt
    x_next = x + V / u_psi * (np.sin(psi_next) - np.sin(psi))
    y_next = y - V / u_psi * (np.cos(psi_next) - np.cos(psi))
    return x_next, y_next, psi_next


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Gimbal:
    """2 軸雲台：θ̇_p = u_p、θ̇_t = u_t（Eq. 2），pan 全向、tilt 有範圍。

    控制律（2024 Eq. 9，細節見前作 Eq. 18–20）：
      期望特徵速度 u̇1 = -λu·u1、v̇1 = -λv·v1，
      u_p = arctan(u̇1 / f)、u_t = arctan(v̇1 / f)。
    前作 Eq. 18 把 arctan(·/f) 解釋成「這一個控制週期要轉的角度」，
    因此這裡的轉速 = 角度 / dt（見 docs/reproduction_log.md）。
    """

    def __init__(self, pan, tilt, params):
        self.pan = float(pan)
        self.tilt = float(tilt)
        self.lam_u, self.lam_v = params["gimbal_lambda"]
        self.tilt_min, self.tilt_max = params["gimbal_tilt_range"]
        self.rate_max = params["gimbal_rate_max"]
        self.acc_max = params.get("gimbal_acc_max") or 0.0     # rad/s²；0 = 不限（舊行為）
        self.rate_act = [0.0, 0.0]
        self.f = params["f"]
        self.dt = params["dt"]

    def rates(self, u1, v1):
        up = np.arctan(-self.lam_u * u1 / self.f) / self.dt
        ut = np.arctan(-self.lam_v * v1 / self.f) / self.dt
        return (np.clip(up, -self.rate_max, self.rate_max),
                np.clip(ut, -self.rate_max, self.rate_max))

    def step(self, u_p, u_t):
        if self.acc_max > 0:   # 真實雲台的角加速度限制（2026-11-22）
            for i, u in enumerate((u_p, u_t)):
                d = float(np.clip(u - self.rate_act[i], -self.acc_max * self.dt, self.acc_max * self.dt))
                self.rate_act[i] = float(np.clip(self.rate_act[i] + d, -self.rate_max, self.rate_max))
            u_p, u_t = self.rate_act
        self.pan = wrap_angle(self.pan + u_p * self.dt)
        self.tilt = float(np.clip(self.tilt + u_t * self.dt, self.tilt_min, self.tilt_max))


class Target:
    """地面目標：等速直線，或由 profile(t) -> (speed, yaw_rate) 指定的複雜運動。"""

    def __init__(self, x, y, heading, speed=0.0, profile=None):
        self.x, self.y, self.heading = float(x), float(y), float(heading)
        self.speed = float(speed)
        self.profile = profile

    def command(self, t):
        if self.profile is None:
            return self.speed, 0.0
        return self.profile(t)

    def step(self, t, dt):
        v, w = self.command(t)
        self.x, self.y, self.heading = unicycle_step(self.x, self.y, self.heading, v, w, dt)
        return v, w

    @property
    def position(self):
        return np.array([self.x, self.y, 0.0])


def fig6_profile(t):
    """Fig. 6b 的目標速度與角速度（從圖上讀出的近似值）。

    0–100 s 等速直線；100–250 s 等速、角速度分段；
    250–400 s 變速直線；400–500 s 變速、角速度分段。
    """
    if t < 100.0:
        return 6.0, 0.0
    if t < 250.0:
        segments = [(100, 120, 0.0), (120, 140, -0.07), (140, 160, 0.0),
                    (160, 180, 0.05), (180, 200, -0.07), (200, 220, 0.05), (220, 250, -0.07)]
        for a, b, w in segments:
            if a <= t < b:
                return 6.0, w
        return 6.0, 0.0
    v = 6.0 + 2.0 * np.sin(2 * np.pi * (t - 250.0) / 21.0)
    if t < 400.0:
        return v, 0.0
    segments = [(400, 410, -0.1), (410, 430, 0.0), (430, 440, 0.1),
                (440, 450, -0.1), (450, 460, 0.1), (460, 500, -0.1)]
    for a, b, w in segments:
        if a <= t < b:
            return v, w
    return v, 0.0
