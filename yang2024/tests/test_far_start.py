"""遠距離起始實驗的設定與指標檢查。"""
import numpy as np

from yang2024 import load_params
from yang2024.far_start import TurnProfile, far_metrics, scenarios

P = load_params()


def test_scenario_grid():
    scs = scenarios(P)
    assert len(scs) == 10
    assert {s["motion"] for s in scs} == {"straight", "circle", "sharp"}
    assert all(s["T"] > (s["d"] - P["R_M"]) / (P["V_t"] - s["Vc"]) for s in scs)


def test_sharp_turn_is_90_degrees():
    prof = TurnProfile(5.0, "sharp", t_turn=10.0)
    dt = 0.01
    heading = sum(prof(k * dt)[1] * dt for k in range(3000))
    assert np.isclose(heading, np.pi / 2, atol=0.01)
    assert prof(0.0) == (5.0, 0.0)


def test_far_metrics_enter_time():
    n = 100
    log = {
        "t": np.arange(n) * P["dt"], "R": np.linspace(300, 50, n),
        "u2": np.zeros(n), "v2": np.zeros(n), "u_psi": np.zeros(n),
        "ideal_valid": np.ones(n, bool), "in_fov": np.ones(n, bool),
        "mode": np.array(["Jg"] * n), "solve_time": np.full(n, 1e-3),
        "solve_ok": np.ones(n, bool),
    }
    m = far_metrics(log, P)
    assert m["entered_RM"]
    k = int(np.argmax(log["R"] <= P["R_M"]))
    assert np.isclose(m["t_enter_s"], log["t"][k])
    assert m["outside_ratio_after_enter"] == 0.0
