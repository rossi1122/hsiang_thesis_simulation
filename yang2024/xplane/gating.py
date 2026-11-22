"""YOLO 量測的合理性檢查（量測閘門）。

2026-11-22 X-Plane 實驗：
- YOLO 從第一張畫面就框到固定在螢幕左上角的東西，雲台與 MPC 一路被帶走；
- 右轉後 YOLO 錯框一張（偏 600 px），雲台以 166 °/s 甩開、目標出畫面而跟丟。
程式原本完全相信 YOLO；這裡用「預測的目標方向」檢查每一筆偵測（只用影像與已知的自身運動，不做定位）：

1. 偵測方向和預測方向的夾角 ≤ 門檻 → 採用。門檻隨跟丟時間放寬（預測越久越不準）。
2. 超過門檻 → 不採用（當作偵測不到，改用推算）。
3. 例外（重新抓回）：已經跟丟 ≥ confirm_after_s，且同一個新位置連續 confirm_n 張都出現 → 採用。
   但如果這段期間相機轉了、偵測的像素位置卻幾乎沒動 → 是固定在螢幕上的東西，不採用。
"""
import math

import numpy as np


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class MeasurementGate:
    def __init__(self, gate_deg=8.0, growth_deg_per_s=3.0, gate_max_deg=20.0, confirm_n=6,
                 confirm_after_s=1.0, confirm_deg=3.0, static_px=5.0, static_rot_deg=2.0):
        self.gate_deg, self.growth, self.gate_max = gate_deg, growth_deg_per_s, gate_max_deg
        self.confirm_n, self.confirm_after_s, self.confirm_deg = confirm_n, confirm_after_s, confirm_deg
        self.static_px, self.static_rot_deg = static_px, static_rot_deg
        self._cand = None          # (los, pixel, cam_axis, 次數)
        self.last_angle = float("nan")
        self.last_reason = ""

    @classmethod
    def from_config(cls, cfg):
        return cls(**{k: v for k, v in (cfg or {}).items() if k != "enabled"})

    def threshold(self, lost_s):
        return min(self.gate_max, self.gate_deg + self.growth * max(0.0, lost_s))

    def check(self, los_meas, los_pred, lost_s, pixel=None, cam_axis=None):
        """回傳 True = 採用這筆偵測。los_*：世界座標的視線向量；cam_axis：相機光軸（世界座標）。"""
        ang = angle_deg(los_meas, los_pred)
        self.last_angle = ang
        if ang <= self.threshold(lost_s):
            self._cand = None
            self.last_reason = "ok"
            return True
        # 不在門檻內：看是不是「真的跑到新位置」的重新抓回
        if self._cand is not None and angle_deg(los_meas, self._cand[0]) <= self.confirm_deg:
            los0, px0, ax0, n = self._cand
            n += 1
            static = (pixel is not None and px0 is not None and cam_axis is not None and ax0 is not None
                      and math.hypot(pixel[0] - px0[0], pixel[1] - px0[1]) < self.static_px
                      and angle_deg(cam_axis, ax0) > self.static_rot_deg)
            if static:
                self._cand = None
                self.last_reason = "static"
                return False
            self._cand = (los_meas, px0, ax0, n)
        else:
            self._cand = (los_meas, pixel, cam_axis, 1)
        if lost_s >= self.confirm_after_s and self._cand[3] >= self.confirm_n:
            self._cand = None
            self.last_reason = "reacquired"
            return True
        self.last_reason = "rejected"
        return False


def rate_limit(rate_prev, rate_cmd, rate_max, acc_max, dt):
    """雲台角速度：先限制角加速度，再限制角速度（rad/s、rad/s²）。acc_max ≤ 0 表示不限。"""
    r = float(rate_cmd)
    if acc_max and acc_max > 0:
        r = rate_prev + float(np.clip(r - rate_prev, -acc_max * dt, acc_max * dt))
    return float(np.clip(r, -rate_max, rate_max))
