"""影像運動拆解用到的公式檢查。"""
import numpy as np

from yang2024 import geometry as g, load_params
from yang2024.motion_budget import target_term_ideal, yaw_column

P = load_params()
F, H = P["f"], P["H"]
ALPHA = g.alpha_from(H, P["R_d"])


def _s2(target, uav, psi):
    u, v, _ = g.project(target, uav, psi, -np.pi / 2, -ALPHA, F)
    return np.array([u, v])


def test_target_term_matches_finite_difference():
    """④ 目標移動造成的 ṡ2 = 目標動一點點時 s2 的變化率（飛機不動）。"""
    uav = np.array([0.0, 0.0, H])
    for psi, target, vt in [(0.3, (20.0, -60.0), (5.0, 2.0)), (-1.0, (-30.0, -90.0), (-7.0, 4.0))]:
        tgt = uav + g.body_to_world(psi) @ np.array([target[0], target[1], -H])
        h = 1e-5
        s0 = _s2(tgt, uav, psi)
        s1 = _s2(tgt + h * np.array([vt[0], vt[1], 0.0]), uav, psi)
        numeric = (s1 - s0) / h
        model = target_term_ideal(s0[0], s0[1], psi, vt[0], vt[1], ALPHA, F, H)
        assert np.allclose(numeric, model, rtol=1e-4, atol=1e-3)


def test_yaw_column_is_eq8():
    u, v = 120.0, -80.0
    du, dv = g.image_model(u, v, 1.0, 0.0, ALPHA, F, 100.0)
    assert np.allclose(yaw_column(u, v, ALPHA, F), [du, dv])
