"""X-Plane 相機模型：內參、像素座標換算、用吉普車真值投影出特徵（Stage 3 用）。

學長環境：X-Plane 全螢幕 1920×1080、橫向視場 60°，mss 擷取 {top: 30, left: 0, 1920×1010}
  f = 960 / tan(30°) ≈ 1662.8 px；主點在 render 中心 (960, 540)，擷取畫面裡是 (960, 510)。
本專案影像座標以主點為原點：u = px − 960、v = py − 510。
"""
import math

from .. import geometry as geo
from .config import camera_f


class XPCamera:
    def __init__(self, cam_cfg):
        self.f = camera_f(cam_cfg)
        self.cx = cam_cfg["render_width"] / 2.0
        self.cy = cam_cfg["render_height"] / 2.0 - cam_cfg["crop_top"]
        self.width = cam_cfg["render_width"]
        self.height = cam_cfg["capture_height"]

    def to_centered(self, px, py):
        return px - self.cx, py - self.cy

    def to_pixel(self, u, v):
        return u + self.cx, v + self.cy

    def in_image(self, u, v):
        px, py = self.to_pixel(u, v)
        return 0 <= px < self.width and 0 <= py < self.height

    def project_truth(self, st, pan, tilt):
        """吉普車真值 → (u1, v1, 深度)。只當作「理想偵測」或拿來和 YOLO 比對，不做定位。"""
        return geo.project(st.jeep_pos, st.pos, st.psi, pan, tilt, self.f,
                           pitch=math.radians(st.pitch_deg), roll=math.radians(st.roll_deg))
