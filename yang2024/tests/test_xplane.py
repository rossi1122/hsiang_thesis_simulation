"""X-Plane 介面的離線檢查（用假的 X-Plane，不需要 X-Plane）。"""
import math

import numpy as np
import pytest

from yang2024 import geometry as geo, load_params
from yang2024.xplane import run_yang, step_test, camera_check
from yang2024.xplane.autopilot import BankToTurnAutopilot
from yang2024.xplane.camera import XPCamera
from yang2024.xplane.config import build, scale_params
from yang2024.xplane.fake_xplane import FakeXPlane
from yang2024.xplane.xp_link import (XPLink, heading_to_psi, local_to_world, psi_to_heading,
                                     world_to_local)

PARAMS, XP, K = build()
PAPER = load_params()


def test_scaling_keeps_dimensionless_numbers():
    p = PARAMS
    for key in ("R_d", "R_M", "dR"):
        assert np.isclose(p[key] / p["H"], PAPER[key] / PAPER["H"])
    # 轉彎半徑 / H、預測時域內飛的距離 / H、目標速度比
    assert np.isclose(p["V_t"] / p["u_max"] / p["H"], PAPER["V_t"] / PAPER["u_max"] / PAPER["H"])
    assert np.isclose(p["V_t"] * p["dt"] * p["N_p"] / p["H"],
                      PAPER["V_t"] * PAPER["dt"] * PAPER["N_p"] / PAPER["H"])
    # 每步可改變的轉彎半徑比例：Δu / u_max 不變
    assert np.isclose(p["a_max"] / p["u_max"], PAPER["a_max"] / PAPER["u_max"])
    assert np.isclose(K["k_T"], 1.6) and np.isclose(K["k_L"], 3.0)
    assert np.isclose(p["f"], 960 / math.tan(math.radians(30)))


def test_scaled_costs_are_equivalent():
    """縮放後同一個無因次狀態的成本相同：q·(k_f·s)² 與 Q_u·(Δu/k_T)² 都不變。"""
    s_paper, du_paper = 100.0, 0.01
    kf, kT = K["k_f"], K["k_T"]
    assert np.isclose(PARAMS["Q_s"][0] * (kf * s_paper) ** 2, PAPER["Q_s"][0] * s_paper**2)
    assert np.isclose(PARAMS["Q_u"] * (du_paper / kT) ** 2, PAPER["Q_u"] * du_paper**2)


def test_steady_state_circle_scaled():
    alpha = geo.alpha_from(PARAMS["H"], PARAMS["R_d"])
    z = geo.ideal_depth(0.0, alpha, PARAMS["f"], PARAMS["H"])
    u = -(PARAMS["f"] / z * PARAMS["V_t"]) / (PARAMS["f"] * np.cos(alpha))
    assert np.isclose(u, -PARAMS["V_t"] / PARAMS["R_d"])  # -0.2 rad/s


def test_coordinate_conversions():
    assert np.isclose(heading_to_psi(0.0), math.pi / 2)        # 北 = +y
    assert np.isclose(heading_to_psi(90.0), 0.0)               # 東 = +x
    for h in (0.0, 45.0, 177.0, 300.0):
        assert np.isclose(psi_to_heading(heading_to_psi(h)), h)
    assert local_to_world(*world_to_local(1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


def test_link_reads_turn_direction():
    fake = FakeXPlane()
    link = XPLink(XP, client=fake, clock=fake.now)
    fake.phi = 30.0  # 右翼下沉 → 右轉
    fake.sleep(0.5)
    st = link.read()
    assert st.psi_rate < 0               # 本專案慣例：右轉為負
    assert st.jeep_ok


def test_gimbal_dref_mapping():
    fake = FakeXPlane()
    link = XPLink(XP, client=fake, clock=fake.now)
    link.gimbal(math.radians(10.0), math.radians(-30.0))   # 往左 10°、往下 30°
    assert np.isclose(fake.head["psi"], -10.0)
    assert np.isclose(fake.head["the"], -30.0 + XP["camera"]["head_pitch_bias_deg"])


def test_autopilot_signs():
    fake = FakeXPlane()
    link = XPLink(XP, client=fake, clock=fake.now)
    ap = BankToTurnAutopilot(XP["autopilot"], V_cmd=30.0, H_cmd=300.0)
    st = link.read()
    elev, ail, rud, thr = ap.update(st, +0.2, 0.02)       # 左轉命令
    assert ap.last["phi_cmd"] < 0 and ail < 0
    elev, ail, rud, thr = ap.update(st, -0.2, 0.02)       # 右轉命令
    assert ap.last["phi_cmd"] > 0 and ail > 0
    fake.ly -= 50.0                                       # 比目標高度低 50 m
    elev, *_ = ap.update(link.read(), 0.0, 0.02)
    assert elev > 0
    assert np.isclose(abs(ap.phi_command_deg(PARAMS["u_max"], 30.0)),
                      math.degrees(math.atan(30.0 * PARAMS["u_max"] / 9.81)))


def test_camera_projection_centered_on_jeep():
    fake = FakeXPlane()
    link = XPLink(XP, client=fake, clock=fake.now)
    cam = XPCamera(XP["camera"])
    st = link.read()
    pan, tilt = geo.pointing_angles(st.jeep_pos, st.pos, st.psi,
                                    math.radians(st.pitch_deg), math.radians(st.roll_deg))
    u, v, depth = cam.project_truth(st, pan, tilt)
    assert depth > 0 and abs(u) < 1e-6 and abs(v) < 1e-6
    assert cam.to_pixel(0.0, 0.0) == (960.0, 510.0)


@pytest.mark.parametrize("controller", ["switch", "lsm"])
def test_run_yang_dry_run(tmp_path, controller):
    out = run_yang.main(["--dry-run", "--time", "15", "--controller", controller, "--out", str(tmp_path)])
    assert (out / "log.csv").exists() and (out / "summary.json").exists()


def test_step_test_dry_run(tmp_path, monkeypatch):
    monkeypatch.setitem(XP["step_test"], "steps", [0.2, 0.0])
    out, res = step_test.main(["--dry-run", "--out", str(tmp_path)])
    assert res["tau_median_s"] is not None and 0.05 < res["tau_median_s"] < 2.0


def test_camera_check_dry_run(tmp_path):
    out, rows = camera_check.main(["--dry-run", "--out", str(tmp_path)])
    base = rows[0]
    right = next(r for r in rows if r["pan_offset_deg"] > 0)
    up = next(r for r in rows if r["tilt_offset_deg"] < 0)
    # 雲台對準後等 1 s 才量，期間吉普車照樣在開（暫停只停飛機）→ 不會剛好在正中央，但要很接近
    assert abs(base["pred_px"] - 960.0) < 120 and abs(base["pred_py"] - 510.0) < 120
    assert right["pred_px"] > base["pred_px"]     # 雲台往左 → 目標在畫面右邊
    assert up["pred_py"] < base["pred_py"]        # 雲台往下 → 目標在畫面上方


def test_scale_params_roundtrip():
    p, k = scale_params(PAPER, PAPER["V_t"], PAPER["H"], PAPER["f"])
    for key in ("u_max", "a_max", "dt", "R_d", "Q_u"):
        assert np.isclose(np.asarray(p[key]), np.asarray(PAPER[key])).all()
