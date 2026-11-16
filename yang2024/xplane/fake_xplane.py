"""假的 X-Plane（只給離線測試用）：介面和 XPlaneConnect 相同，飛機動態非常簡化。

用途：在沒有 X-Plane 的地方（雲端、CI）把 run_yang / step_test / camera_check 從頭跑到尾，
確認程式路徑、座標與符號一致。它**不能**代表 SP80 的真實飛行特性，增益一定要在 X-Plane 裡調。
符號假設（和 X-Plane 一致）：副翼正 = 右滾、升降舵正 = 抬頭、方向舵正 = 機頭往右、真航向順時針。
"""
import math

G = 9.81


class FakeXPlane:
    def __init__(self, jeep_speed=13.0, jeep_heading_deg=177.0, ground_y=200.0):
        self.t = 0.0
        self.paused = False
        self.ctrl = [0.0, 0.0, 0.0, 0.45]
        self.ground_y = ground_y
        # 飛機（X-Plane local 座標）
        self.lx, self.ly, self.lz = 0.0, ground_y + 300.0, 0.0
        self.hdg, self.theta, self.phi = 0.0, 2.0, 0.0
        self.V, self.P, self.q_pitch = 30.0, 0.0, 0.0
        self.hdg_rate = 0.0
        # 吉普車
        self.jx, self.jz, self.jhdg, self.jv = 0.0, 0.0, jeep_heading_deg, jeep_speed
        self.head = {}

    # ---------------------------------------------------------- 時間
    def now(self):
        return self.t

    def sleep(self, dt):
        if dt <= 0:
            return
        n = max(1, int(math.ceil(dt / 0.005)))
        h = dt / n
        for _ in range(n):
            if not self.paused:
                self._step(h)
            self.t += h

    def _step(self, h):
        elev, ail, rud, thr = self.ctrl
        self.P += (150.0 * ail - self.P) * h / 0.2          # deg/s
        self.phi = max(-70.0, min(70.0, self.phi + self.P * h))
        self.hdg_rate = math.degrees(G * math.tan(math.radians(self.phi)) / max(self.V, 1.0))
        self.hdg = (self.hdg + self.hdg_rate * h) % 360.0
        self.q_pitch += (40.0 * elev - self.q_pitch) * h / 0.2
        self.theta = max(-30.0, min(30.0, self.theta + self.q_pitch * h))
        gamma = math.radians(self.theta - 2.0) - 0.002 * (1 / math.cos(math.radians(self.phi)) - 1) * 50
        self.V += ((thr - 0.45) * 3.0 - G * math.sin(gamma) * 0.3) * h
        hr = math.radians(self.hdg)
        self.lx += self.V * math.sin(hr) * h
        self.lz -= self.V * math.cos(hr) * h
        self.ly += self.V * math.sin(gamma) * h
        jr = math.radians(self.jhdg)
        self.jx += self.jv * math.sin(jr) * h
        self.jz -= self.jv * math.cos(jr) * h

    # ---------------------------------------------------------- XPC 介面
    def _values(self):
        hr, phr = math.radians(self.hdg), math.radians(self.phi)
        gamma = math.radians(self.theta - 2.0)
        R = self.hdg_rate * math.cos(phr)
        Q = self.q_pitch + self.hdg_rate * math.sin(phr)
        pos = "sim/flightmodel/position/"
        mp = "sim/multiplayer/position/plane1_"
        return {
            pos + "local_x": self.lx, pos + "local_y": self.ly, pos + "local_z": self.lz,
            pos + "local_vx": self.V * math.sin(hr), pos + "local_vy": self.V * math.sin(gamma),
            pos + "local_vz": -self.V * math.cos(hr),
            pos + "psi": self.hdg, pos + "theta": self.theta, pos + "phi": self.phi,
            pos + "P": self.P, pos + "Q": Q, pos + "R": R,
            pos + "true_airspeed": self.V, pos + "y_agl": self.ly - self.ground_y,
            mp + "x": self.jx, mp + "y": self.ground_y + 1.0, mp + "z": self.jz, mp + "psi": self.jhdg,
        }

    def getDREFs(self, drefs):
        vals = self._values()
        return [(vals[d],) if d in vals else () for d in drefs]

    def getDREF(self, dref):
        return self.getDREFs([dref])[0]

    def sendCTRL(self, values, ac=0):
        for i, v in enumerate(values[:4]):
            if v != -998:
                self.ctrl[i] = float(v)

    def sendDREF(self, dref, value):
        self.sendDREFs([dref], [value])

    def sendDREFs(self, drefs, values):
        pos = "sim/flightmodel/position/"
        for d, v in zip(drefs, values):
            if d == pos + "local_x":
                self.lx = v
            elif d == pos + "local_y":
                self.ly = v
            elif d == pos + "local_z":
                self.lz = v
            elif d == pos + "psi":
                self.hdg = v
            elif d == pos + "theta":
                self.theta = v
            elif d == pos + "phi":
                self.phi = v
            elif d == pos + "P":
                self.P = v
            elif d == pos + "local_vx":
                self._vx = v
            elif d == pos + "local_vz":
                self._vz = v
                self.V = max(10.0, math.hypot(getattr(self, "_vx", 0.0), v))
            elif d.startswith("sim/graphics/view/pilots_head"):
                self.head[d.rsplit("_", 1)[-1]] = v

    def pauseSim(self, pause):
        self.paused = bool(pause)

    def sendTEXT(self, msg, x=-1, y=-1):
        pass


class RealClock:
    def __init__(self):
        import time
        self._time = time

    def now(self):
        return self._time.time()

    def sleep(self, dt):
        if dt > 0:
            self._time.sleep(dt)
