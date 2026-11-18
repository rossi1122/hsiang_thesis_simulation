"""雲台模型（2026-11-18）。

body   ：Yang 2024 原版。兩軸（pan、tilt），角度「相對機身」，沒有增穩。
         飛機一傾斜相機就跟著歪；tilt 範圍用論文 HITL 的 [-1.55, 0.17] rad。
follow ：SIYI A8 mini 的「跟隨模式」。三軸增穩：
         - 指令 (pan, tilt) 定義在「水平航向座標」：pan = 相對機頭水平方向的方位、tilt = 對地平線的俯角；
         - 雲台自動抵消飛機的俯仰與滾轉，相機不會跟著飛機歪；
         - 機構是 yaw（底座、繞機體 z）→ roll → pitch（相機），每個關節有規格限制，
           官方手冊：yaw ±160°、pitch −135°～+45°、roll ±30°。超出時夾在極限，殘餘的姿態就會漏到畫面上（和實機一樣）。
         - 目標在正後方（yaw 超過 ±160°）時，會改用「pitch 翻過 −90°」的那一組關節（畫面上下顛倒，實機也是這樣）。
"""
import math

import numpy as np

from . import geometry as geo

FLIP = np.diag([-1.0, -1.0, 1.0])   # 同一條光軸、影像轉 180°


class GimbalModel:
    def __init__(self, mode="body", yaw_limit_deg=160.0, pitch_limit_deg=(-135.0, 45.0), roll_limit_deg=30.0,
                 body_tilt_range=(-1.55, 0.17)):
        if mode not in ("body", "follow"):
            raise ValueError("gimbal mode 必須是 body 或 follow")
        self.mode = mode
        self.yaw_lim = math.radians(yaw_limit_deg)
        self.pitch_lim = (math.radians(pitch_limit_deg[0]), math.radians(pitch_limit_deg[1]))
        self.roll_lim = math.radians(roll_limit_deg)
        self.body_tilt_range = body_tilt_range
        self.prev_joints = None

    @classmethod
    def from_config(cls, gcfg, params, mode=None):
        return cls(mode=mode or gcfg.get("mode", "body"),
                   yaw_limit_deg=gcfg.get("yaw_limit_deg", 160.0),
                   pitch_limit_deg=tuple(gcfg.get("pitch_limit_deg", (-135.0, 45.0))),
                   roll_limit_deg=gcfg.get("roll_limit_deg", 30.0),
                   body_tilt_range=tuple(params["gimbal_tilt_range"]))

    # ------------------------------------------------------------------
    def _clip(self, j):
        yaw, roll, pitch_up = j
        return (float(np.clip(yaw, -self.yaw_lim, self.yaw_lim)),
                float(np.clip(roll, -self.roll_lim, self.roll_lim)),
                float(np.clip(pitch_up, self.pitch_lim[0], self.pitch_lim[1])))

    def rows(self, pan, tilt, pitch, roll):
        """雲台指令 + 飛機俯仰、滾轉 → (C_H, C_body, info)。

        C_H   ：相機列在水平航向座標（量測換算、真值投影用）
        C_body：相機列在機體座標（換成 X-Plane 頭部角度用）
        info  ：joints = 機構關節 (yaw, roll, pitch_up)、saturated = 有沒有關節碰到極限、
                flipped = 是否用了翻過頭的那一組、cam_roll = 影像相對地平線的滾轉
        """
        if self.mode == "body":
            lo, hi = self.body_tilt_range
            tilt_c = float(np.clip(tilt, lo, hi))
            C_body = geo.camera_rows(pan, tilt_c)
            C_H = geo.rows_body_to_level(C_body, pitch, roll)
            info = {"joints": (pan, 0.0, tilt_c), "saturated": tilt_c != tilt, "flipped": False}
        else:
            C_des_body = geo.rows_level_to_body(geo.camera_rows(pan, tilt), pitch, roll)
            best = None
            for k, C in enumerate((C_des_body, FLIP @ C_des_body)):
                j = geo.joints_from_rows(C)
                jc = self._clip(j)
                viol = abs(geo.wrap(j[0] - jc[0])) + abs(j[1] - jc[1]) + abs(j[2] - jc[2])
                if self.prev_joints is None:
                    dist = float(k)
                else:
                    dist = sum(abs(geo.wrap(a - b)) for a, b in zip(jc, self.prev_joints))
                key = (round(viol, 4), dist)
                if best is None or key < best[0]:
                    best = (key, jc, viol, k)
            _, jc, viol, k = best
            self.prev_joints = jc
            C_body = geo.rows_from_joints(*jc)
            C_H = geo.rows_body_to_level(C_body, pitch, roll)
            info = {"joints": jc, "saturated": viol > 1e-4, "flipped": k == 1}
        right = C_H[0]
        info["cam_roll"] = float(math.asin(float(np.clip(-right[2], -1.0, 1.0))))
        return C_H, C_body, info

    def head_angles(self, C_body):
        """X-Plane 頭部視角 (偏航, 俯仰往上為正, 滾轉)，單位 rad。"""
        return geo.head_angles_from_rows(C_body)
