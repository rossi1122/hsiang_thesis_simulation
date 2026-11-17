"""和 X-Plane 11 溝通（透過學長 repo 裡的 XPlaneConnect，不修改學長的檔案）。

座標換算（X-Plane 的 local OpenGL：local_x 東、local_y 上、local_z 南）
  本專案世界座標  x = local_x（東）、y = -local_z（北）、z = local_y（上）
  航向 ψ（從 +x 逆時針、rad）= π/2 − 真航向（從北順時針、deg → rad）
  滾轉 φ：右翼下沉為正（與 X-Plane 相同）；俯仰 θ：抬頭為正（相同）
  航向變化率 ψ̇ = −(Q·sinφ + R·cosφ)/cosθ（X-Plane 的 P、Q、R 是機體角速度 deg/s）
每一圈只用一次 getDREFs 讀完所有狀態（學長的程式每圈分開讀約 30 次）。
"""
import math
import sys
import time
from dataclasses import dataclass

from .. import REPO_DIR

SENIOR_DIR = REPO_DIR / "模擬程式"

POS = "sim/flightmodel/position/"
STATE_DREFS = [POS + n for n in (
    "local_x", "local_y", "local_z", "local_vx", "local_vy", "local_vz",
    "psi", "theta", "phi", "P", "Q", "R", "true_airspeed", "y_agl")]


def jeep_drefs(index):
    mp = f"sim/multiplayer/position/plane{index}_"
    return [mp + n for n in ("x", "y", "z", "psi")]


def import_senior(module):
    """從學長的 模擬程式/ 匯入模組（只讀取，不修改）。"""
    if str(SENIOR_DIR) not in sys.path:
        sys.path.insert(0, str(SENIOR_DIR))
    return __import__(module, fromlist=["*"])


def heading_to_psi(true_heading_deg):
    return math.pi / 2 - math.radians(true_heading_deg)


def psi_to_heading(psi):
    return (90.0 - math.degrees(psi)) % 360.0


def local_to_world(lx, ly, lz):
    return lx, -lz, ly


def world_to_local(x, y, z):
    return x, z, -y


@dataclass
class State:
    t: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    psi: float          # rad，本專案慣例
    heading_deg: float  # X-Plane 真航向
    pitch_deg: float
    roll_deg: float
    P: float
    Q: float
    R: float            # 機體角速度 deg/s
    airspeed: float     # m/s
    agl: float          # m
    psi_rate: float     # rad/s，本專案慣例（逆時針為正）
    jeep_ok: bool
    jx: float
    jy: float
    jz: float
    jpsi: float

    @property
    def pos(self):
        return (self.x, self.y, self.z)

    @property
    def jeep_pos(self):
        return (self.jx, self.jy, self.jz)


class XPLink:
    def __init__(self, xp, client=None, clock=time.time):
        cfg = xp["link"]
        if client is None:
            xpc = import_senior("xplane_connect.xpc3")
            client = xpc.XPlaneConnect(xpHost=cfg["host"], xpPort=cfg["port"], timeout=cfg["timeout_ms"])
        self.c = client
        self.clock = clock
        self.cam = xp["camera"]
        self.drefs = STATE_DREFS + jeep_drefs(cfg["jeep_index"])

    # ----------------------------------------------------------- 讀
    def read(self):
        rows = self.c.getDREFs(self.drefs)
        v = [r[0] if len(r) else float("nan") for r in rows]
        lx, ly, lz, lvx, lvy, lvz, hdg, th, ph, P, Q, R, tas, agl = v[:14]
        jlx, jly, jlz, jhdg = v[14:18]
        x, y, z = local_to_world(lx, ly, lz)
        vx, vy, vz = local_to_world(lvx, lvy, lvz)
        jx, jy, jz = local_to_world(jlx, jly, jlz)
        phr, thr = math.radians(ph), math.radians(th)
        hdg_rate = (Q * math.sin(phr) + R * math.cos(phr)) / max(math.cos(thr), 1e-3)
        jeep_ok = all(math.isfinite(a) for a in (jlx, jly, jlz)) and (abs(jlx) + abs(jlz)) > 1e-3
        return State(
            t=self.clock(), x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
            psi=heading_to_psi(hdg), heading_deg=hdg, pitch_deg=th, roll_deg=ph,
            P=P, Q=Q, R=R, airspeed=tas, agl=agl, psi_rate=-math.radians(hdg_rate),
            jeep_ok=jeep_ok, jx=jx, jy=jy, jz=jz,
            jpsi=heading_to_psi(jhdg) if math.isfinite(jhdg) else float("nan"),
        )

    # ----------------------------------------------------------- 寫
    def controls(self, elevator, aileron, rudder, throttle):
        """順序同學長的 send_ctrl：[升降舵, 副翼, 方向舵, 油門]。"""
        self.c.sendCTRL([float(elevator), float(aileron), float(rudder), float(throttle)])

    def gimbal(self, pan, tilt):
        """pan 往左為正、tilt 往下為負（rad）→ pilots_head_psi / the（deg，含學長的 8.5° 校正）。"""
        head_psi = self.cam["head_psi_sign"] * math.degrees(pan)
        head_the = math.degrees(tilt) + self.cam["head_pitch_bias_deg"]
        self.c.sendDREFs(["sim/graphics/view/pilots_head_psi", "sim/graphics/view/pilots_head_the"],
                         [head_psi, head_the])

    def pause(self, yes=True):
        self.c.pauseSim(bool(yes))

    def message(self, text):
        try:
            self.c.sendTEXT(text)
        except Exception:  # noqa: BLE001 — 只是顯示用
            pass

    def place(self, x, y, z, psi, speed, throttle=0.45):
        """把飛機放到世界座標 (x, y, z)、航向 psi、空速 speed，並暫停（照學長 reset() 的步驟）。"""
        c = self.c
        c.pauseSim(True)
        c.sendDREF("sim/operation/override/override_joystick", 1)
        c.sendCTRL([0, 0, 0, 0])
        c.sendDREF("sim/flightmodel/controls/parkbrake", 0)
        zero = ["theta", "phi", "local_vx", "local_vy", "local_vz", "local_ax", "local_ay", "local_az",
                "Prad", "Qrad", "Rrad", "q", "groundspeed", "indicated_airspeed", "indicated_airspeed2",
                "true_airspeed", "M", "N", "L", "P", "Q", "R", "P_dot", "Q_dot", "R_dot"]
        c.sendDREFs([POS + n for n in zero], [0.0] * len(zero))
        lx, ly, lz = world_to_local(x, y, z)
        c.sendDREFs([POS + "phi", POS + "theta", POS + "psi"], [0.0, 2.0, psi_to_heading(psi)])
        c.sendDREFs([POS + "local_x", POS + "local_y", POS + "local_z"], [lx, ly, lz])
        c.sendDREFs(["sim/operation/fix_all_systems"], [1])
        c.sendDREF("sim/flightmodel/engine/ENGN_mixt", 0.61)
        c.sendDREFs(["sim/flightmodel/weight/m_fuel1", "sim/flightmodel/weight/m_fuel2"], [232, 232])
        vx, vy = speed * math.cos(psi), speed * math.sin(psi)
        lvx, _, lvz = world_to_local(vx, vy, 0.0)
        c.sendDREFs([POS + "local_vx", POS + "local_vy", POS + "local_vz"], [lvx, 0.0, lvz])
        c.sendCTRL([0.0, 0.0, 0.0, throttle])
        c.sendDREFs(["sim/graphics/view/pilots_head_x", "sim/graphics/view/pilots_head_y",
                     "sim/graphics/view/pilots_head_z"], [0.0, 0.0, 0.0])


def place_relative_to_jeep(link, st, xp, k, H, jeep_shift=(0.0, 0.0)):
    """依 scenario 把飛機放在吉普車附近（吉普車座標：x 沿前進方向、y 往左）。回傳 (x, y, z, psi)。

    jeep_shift：吉普車在飛機擺好、暫停等待的這段時間會再開走的位移（世界座標），
    先加上去，飛機解除暫停時和吉普車的距離才會是設定值（2026-11-17 第一次閉迴路：沒補時起始距離 280 m 而非 255 m）。
    """
    sc = xp["scenario"]
    if sc["start"] == "far":
        dx, dy, dpsi = -sc["far_distance_paper"] * k["k_L"], 0.0, 0.0
    else:  # paper：論文 (-85, 0)、航向和目標垂直往左，長度乘 k_L
        dx, dy, dpsi = -85.0 * k["k_L"], 0.0, math.pi / 2
    c, s = math.cos(st.jpsi), math.sin(st.jpsi)
    x = st.jx + jeep_shift[0] + c * dx - s * dy
    y = st.jy + jeep_shift[1] + s * dx + c * dy
    z = st.jz + H
    psi = st.jpsi + dpsi
    link.place(x, y, z, psi, xp["scaling"]["V_t"], throttle=xp["autopilot"]["thr_trim"])
    return x, y, z, psi
