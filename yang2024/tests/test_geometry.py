"""確認座標慣例和論文式子一致。執行：python -m pytest yang2024/tests -q"""
import numpy as np
import pytest

from yang2024 import geometry as g, load_params

P = load_params()
F, H, V = P["f"], P["H"], P["V_t"]
ALPHA = g.alpha_from(H, P["R_d"])


def _ideal_feature(target, uav, psi):
    """直接用 Ideal State 相機（θp = -π/2、θt = -α）投影。"""
    u, v, _ = g.project(target, uav, psi, -np.pi / 2, -ALPHA, F)
    return np.array([u, v])


@pytest.mark.parametrize("u_psi", [-0.4, -0.1, 0.0, 0.3])
@pytest.mark.parametrize("offset", [(0.0, -50.0), (20.0, -70.0), (-30.0, -40.0)])
def test_eq8_matches_finite_difference(u_psi, offset):
    """Eq. 8 = 相機前進 + 偏航時特徵的真實變化率（靜止目標）。"""
    uav = np.array([0.0, 0.0, H])
    psi = 0.3
    # offset 是機體座標的水平位置（x 前、y 左），目標在右側 → y < 0
    target = uav + g.body_to_world(psi) @ np.array([offset[0], offset[1], -H])
    s = _ideal_feature(target, uav, psi)
    h = 1e-5
    uav_next = uav + h * V * np.array([np.cos(psi), np.sin(psi), 0.0])
    s_next = _ideal_feature(target, uav_next, psi + h * u_psi)
    numeric = (s_next - s) / h
    z = g.ideal_depth(s[1], ALPHA, F, H)
    model = np.array(g.image_model(s[0], s[1], u_psi, V, ALPHA, F, z))
    assert np.allclose(numeric, model, rtol=1e-4, atol=1e-3)


def test_steady_state_circling():
    """s2 = 0 ⇔ 目標在右側 R_d、u_psi = -V/R_d（右轉繞目標）。"""
    uav = np.array([0.0, 0.0, H])
    target = np.array([0.0, -P["R_d"], 0.0])
    s = _ideal_feature(target, uav, 0.0)
    assert np.allclose(s, 0.0, atol=1e-9)
    z = g.ideal_depth(0.0, ALPHA, F, H)
    # u̇2 = A·u_psi + f/z·V = 0
    u_psi = -(F / z * V) / (F * np.cos(ALPHA))
    assert np.isclose(u_psi, -V / P["R_d"])


def test_to_ideal_is_pure_rotation():
    """任意雲台角度量到的特徵，轉到 Ideal State 要和直接用 Ideal 相機投影一致。"""
    rng = np.random.default_rng(0)
    uav = np.array([10.0, -5.0, H])
    for _ in range(20):
        psi = rng.uniform(-np.pi, np.pi)
        target = uav + np.array([rng.uniform(-80, 80), rng.uniform(-80, 80), -H])
        pan, tilt = g.pointing_angles(target, uav, psi)
        pan += rng.uniform(-0.1, 0.1)
        tilt += rng.uniform(-0.1, 0.1)
        u1, v1, depth = g.project(target, uav, psi, pan, tilt, F)
        assert depth > 0
        u2, v2 = g.to_ideal(u1, v1, pan, tilt, ALPHA, F)
        assert np.allclose([u2, v2], _ideal_feature(target, uav, psi), atol=1e-6)


def test_eq54_equals_down_camera():
    """Eq. 54 的 (u3, v3) = 垂直朝下相機的投影，且 g1 = 0 剛好在水平距離 R_M。"""
    uav = np.array([0.0, 0.0, H])
    for bx, by in [(0.0, -50.0), (30.0, -90.0), (-40.0, -20.0)]:
        target = np.array([bx, by, 0.0])
        u2, v2 = _ideal_feature(target, uav, 0.0)
        u3, v3 = g.ideal_to_down(u2, v2, ALPHA, F)
        u3d, v3d, _ = g.project(target, uav, 0.0, -np.pi / 2, -np.pi / 2, F)
        assert np.allclose([u3, v3], [u3d, v3d], atol=1e-6)
        R = np.hypot(bx, by)
        assert np.isclose(g.perception_g(u3, v3, F, R, H), 0.0, atol=1e-9)


def test_ideal_depth():
    """幾何深度 = 目標在 Ideal 相機光軸方向的距離。"""
    uav = np.array([0.0, 0.0, H])
    target = np.array([25.0, -70.0, 0.0])
    u2, v2 = _ideal_feature(target, uav, 0.0)
    z_true = (g.camera_rows(-np.pi / 2, -ALPHA) @ (target - uav))[2]
    assert np.isclose(g.ideal_depth(v2, ALPHA, F, H), z_true)


def test_ideal_and_down_matches_eq54_and_flags_invalid():
    """有效時 (u3, v3) 與 Eq. 54 相同；目標在左側很遠時 (u2, v2) 標為無效。"""
    uav = np.array([0.0, 0.0, H])
    for target, expect_valid in [((300.0, 0.0, 0.0), True), ((30.0, -80.0, 0.0), True),
                                 ((0.0, 400.0, 0.0), False)]:
        pan, tilt = g.pointing_angles(target, uav, 0.0)
        u1, v1, _ = g.project(target, uav, 0.0, pan, tilt, F)
        u2, v2, u3, v3, valid = g.ideal_and_down(u1, v1, pan, tilt, ALPHA, F)
        assert valid == expect_valid
        u3d, v3d, _ = g.project(target, uav, 0.0, -np.pi / 2, -np.pi / 2, F)
        assert np.allclose([u3, v3], [u3d, v3d], atol=1e-6)
        if valid:
            assert np.allclose([u3, v3], g.ideal_to_down(u2, v2, ALPHA, F), atol=1e-6)
