"""MPC、切換邏輯、閉迴路的基本檢查。"""
import numpy as np
import pytest

from yang2024 import load_params
from yang2024.baselines import lsm_u_psi
from yang2024.geometry import alpha_from
from yang2024.metrics import summarize
from yang2024.mpc import Switcher, YangMPC
from yang2024.sim import simulate

P = load_params()


@pytest.fixture(scope="module")
def mpc():
    return YangMPC(P)


def test_warm_start_respects_limits(mpc):
    U = mpc.lsm_warm_start(150.0, -120.0, 0.0)
    assert np.all(np.abs(U) <= P["u_max"] + 1e-12)
    dU = np.diff(np.concatenate([[0.0], U]))
    assert np.all(np.abs(dU) <= P["a_max"] + 1e-12)


@pytest.mark.parametrize("mode", ["J1", "J2", "Jg"])
def test_solution_respects_limits(mpc, mode):
    u_prev = -0.3
    U0 = np.full(P["N_p"], u_prev)
    U, info = mpc.solve(mode, 120.0, -80.0, u_prev, U0)
    assert info["success"], info["status"]
    assert np.all(np.abs(U) <= P["u_max"] + 1e-8)
    dU = np.diff(np.concatenate([[u_prev], U]))
    assert np.all(np.abs(dU) <= P["a_max"] + 1e-8)


def test_lsm_steady_state():
    """s2 = 0 時 LSM 輸出 -V/(z_d·cosα)（前作的常數深度）。"""
    alpha = alpha_from(P["H"], P["R_d"])
    u = lsm_u_psi(0.0, 0.0, P["V_t"], alpha, P["f"], P)
    assert np.isclose(u, -P["V_t"] / (P["lsm_depth"] * np.cos(alpha)))


@pytest.mark.parametrize("mapping,expected", [
    ("text", ("Jg", "J2", "J1")),
    ("proof", ("Jg", "J1", "J2")),
    ("literal", ("J1", "J2", "Jg")),
])
def test_switch_mapping(mapping, expected):
    p = dict(P, switch_mapping=mapping, n_max=3)
    sw = Switcher(p, g1_initial=0.5)  # 一開始在範圍外 → n1 = 0
    seen = [sw.mode]
    for _ in range(4):
        sw.update(-0.1)
        seen.append(sw.mode)
    assert seen[0] == expected[0]
    assert seen[1] == expected[1]
    assert seen[-1] == expected[2]
    sw.update(0.2)
    assert sw.n1 == 0


def test_short_closed_loop_runs_clean(mpc):
    log = simulate(P, "switch", Vc=7.0, T=5.0, mpc=mpc)
    m = summarize(log, P)
    assert m["solve_failures"] == 0
    assert np.all(log["visible"])
