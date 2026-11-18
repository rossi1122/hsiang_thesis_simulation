"""增穩雲台模型與一般化相機姿態的測試（2026-11-18）。"""
import math

import numpy as np
import pytest

from yang2024 import geometry as geo
from yang2024.gimbal_model import GimbalModel

RNG = np.random.default_rng(0)
F = 1662.77
ALPHA = geo.alpha_from(300.0, 150.0)


def test_joints_reduce_to_two_axis():
    for _ in range(50):
        p, t = RNG.uniform(-3, 3), RNG.uniform(-1.5, 0.5)
        assert np.allclose(geo.rows_from_joints(p, 0.0, t), geo.camera_rows(p, t))
        assert np.allclose(geo.head_angles_from_rows(geo.camera_rows(p, t)), (p, t, 0.0), atol=1e-9)


def test_joint_and_head_round_trip():
    for _ in range(100):
        j = (RNG.uniform(-3, 3), RNG.uniform(-1.2, 1.2), RNG.uniform(-1.4, 1.4))
        C = geo.rows_from_joints(*j)
        assert np.allclose(geo.joints_from_rows(C), j, atol=1e-9)
        a, pu, r = geo.head_angles_from_rows(C)
        C2 = geo.M0 @ (geo.rot_z(a) @ geo.rot_y(-pu) @ geo.rot_x(r)).T   # X-Plane：先偏航、再俯仰、最後滾轉
        assert np.allclose(C2, C, atol=1e-9)


def test_generalized_functions_match_original():
    """新的「相機列」寫法和原本的 ideal_and_down / project 完全一致（原版實驗結果不變）。"""
    for _ in range(50):
        pan, tilt = RNG.uniform(-3, 3), RNG.uniform(-1.4, 0.1)
        pitch, roll = RNG.uniform(-0.2, 0.2), RNG.uniform(-0.8, 0.8)
        u1, v1 = RNG.uniform(-900, 900), RNG.uniform(-500, 500)
        C_H = geo.rows_body_to_level(geo.camera_rows(pan, tilt), pitch, roll)
        a = geo.ideal_and_down(u1, v1, pan, tilt, ALPHA, F, pitch=pitch, roll=roll)
        b = geo.ideal_and_down_from_ray(geo.ray_level_from_pixel(u1, v1, C_H, F), ALPHA, F)
        assert a[4] == b[4]
        assert np.allclose([x for x in a[:4]], [x for x in b[:4]], equal_nan=True)
        tgt, uav, psi = RNG.uniform(-500, 500, 3) * [1, 1, 0], [0.0, 0.0, 300.0], RNG.uniform(-3, 3)
        assert np.allclose(geo.project(tgt, uav, psi, pan, tilt, F, pitch=pitch, roll=roll),
                           geo.project_rows(tgt, uav, psi, C_H, F))


def test_ideal_ray_inverse():
    for u2, v2 in [(0, 0), (-600, 300), (400, -200)]:
        out = geo.ideal_and_down_from_ray(geo.ideal_ray_level(u2, v2, F, ALPHA), ALPHA, F)
        assert np.allclose(out[:2], (u2, v2))


def test_follow_mode_stabilizes_bank_to_the_side():
    """看右下方（盤旋時的情況），飛機坡度 ±46°、俯仰 ±8°：相機在水平座標完全不動、影像不傾斜。"""
    g = GimbalModel("follow")
    for roll in np.radians([-46, -30, 0, 30, 46]):
        for pitch in np.radians([-8, 0, 8]):
            C_H, _, info = g.rows(-math.pi / 2, -math.radians(60), pitch, roll)
            assert not info["saturated"]
            assert np.allclose(C_H, geo.camera_rows(-math.pi / 2, -math.radians(60)), atol=1e-9)
            assert abs(info["cam_roll"]) < 1e-9


def test_follow_mode_roll_joint_limit():
    """看正前方時飛機坡度 45° 要靠 roll 關節抵消，但 SIYI 只有 ±30° → 殘留約 15° 的影像傾斜。"""
    g = GimbalModel("follow")
    _, _, info = g.rows(0.0, -math.radians(30), 0.0, math.radians(45))
    assert info["saturated"]
    assert abs(math.degrees(info["joints"][1])) == pytest.approx(30.0)
    assert 10.0 < abs(math.degrees(info["cam_roll"])) < 20.0


def test_e1_failure_case():
    """2026-11-18 YOLO 版 E1：飛機往左傾 46°、目標在右下方（俯角 60°）。
    兩軸機身雲台：在機體座標裡目標已經「過了機腹正下方」（60° + 46° > 90°），要繼續對準只能把 pan 甩 180° 到左邊
    → 實際 log 中就是 tilt 撞到 −89°、pan 以 172°/s 亂甩。增穩雲台：pan 留在右邊、沒有任何關節碰到極限、目標在畫面中央。"""
    uav, psi, pitch = np.array([0.0, 0.0, 300.0]), 0.0, 0.0
    target = np.array([0.0, -300.0 / math.tan(math.radians(60)), 0.0])     # 正右方、俯角 60°
    pan_20, _ = geo.pointing_angles(target, uav, psi, pitch, math.radians(-20))
    pan_46, tilt_46 = geo.pointing_angles(target, uav, psi, pitch, math.radians(-46))
    assert pan_20 < 0 and pan_46 > 0                                          # 坡度從 −20° 到 −46°：pan 從右邊翻到左邊
    assert abs(geo.wrap(pan_46 - pan_20)) > math.radians(170)
    follow = GimbalModel("follow")
    az, el = geo.los_angles(geo.rot_z(psi).T @ (target - uav))
    for roll in np.radians([-20, -46]):
        C_H, _, info = follow.rows(az, el, pitch, roll)
        assert not info["saturated"] and not info["flipped"]
        assert info["joints"][0] < 0                                          # yaw 關節一直在右邊
        u, v, _ = geo.project_rows(target, uav, psi, C_H, F)
        assert abs(u) < 1e-6 and abs(v) < 1e-6


def test_follow_mode_target_behind_uses_flip():
    """目標在正後方（yaw 超出 ±160°）：改用 pitch 翻過 −90° 的那一組，仍然對得準。"""
    g = GimbalModel("follow")
    uav = np.array([0.0, 0.0, 300.0])
    target = np.array([-200.0, 10.0, 0.0])                                    # 正後方偏左一點、俯角約 56°
    az, el = geo.los_angles(target - uav)
    C_H, _, info = g.rows(az, el, 0.0, 0.0)
    assert info["flipped"] and not info["saturated"]
    assert info["joints"][2] < -math.pi / 2
    u, v, _ = geo.project_rows(target, uav, 0.0, C_H, F)
    assert abs(u) < 1e-6 and abs(v) < 1e-6
