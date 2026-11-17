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


def test_profile_original_is_paper_values():
    """original 版：k 全部 = 1，MPC 參數就是 Table 1（只有 f 換成 X-Plane 的）。"""
    from yang2024 import load_params
    from yang2024.xplane.config import build
    params, xp, k = build(profile="original")
    paper = load_params()
    assert xp["profile"] == "original"
    assert np.isclose(k["k_L"], 1) and np.isclose(k["k_V"], 1) and np.isclose(k["k_T"], 1)
    for key in ("V_t", "H", "R_d", "R_M", "dR", "u_max", "a_max", "dt", "Q_u"):
        assert np.isclose(params[key], paper[key]), key
    scaled, xs, ks = build(profile="scaled")
    assert np.isclose(scaled["V_t"], 30) and np.isclose(ks["k_L"], 3)
    # 兩個版本的共用設定（相機、自駕儀）一樣
    assert xs["camera"] == xp["camera"]


def test_step_commands_follow_u_max():
    from yang2024.xplane.step_test import step_commands
    cfg = XP["step_test"]
    assert np.isclose(max(step_commands(cfg, 0.3125)), 0.3125)
    assert np.isclose(max(step_commands(cfg, 0.5)), 0.5)
    assert np.isclose(step_commands(cfg, 0.3125)[0], 0.1)


def test_run_yang_original_dry_run(tmp_path):
    out = run_yang.main(["--dry-run", "--profile", "original", "--time", "20", "--out", str(tmp_path)])
    import json
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["summary"]
    assert s["profile"] == "original" and np.isclose(s["scale"]["k_L"], 1)


def test_yolo_stats():
    """YOLO 統計：延遲、更新頻率、和真值的像素誤差。"""
    from yang2024.xplane.common import Recorder
    from yang2024.xplane.run_yang import yolo_stats
    rec = Recorder()
    for i in range(100):
        t = i * 0.08
        ts = (i // 2) * 0.16 - 0.05          # 每 0.16 s 一張、拍攝在 0.05 s 前
        rec.add(t=t, mpc_tick=1, yolo_ts=ts, detected=1.0, u1=3.0, v1=4.0, u1_truth=0.0, v1_truth=0.0)
    s = yolo_stats(rec, np.ones(100, bool))
    assert abs(s["yolo_hz"] - 1 / 0.16) < 0.1
    assert 50 <= s["yolo_age_ms_mean"] <= 130
    assert np.isclose(s["yolo_err_px_rms"], 5.0)


def test_run_yang_yolo_path_with_fake_detector(tmp_path, monkeypatch):
    """用假的 vision_driver 走一次 YOLO 流程：暖機 → 等第一次偵測 → 閉迴路 → summary 有 YOLO 統計。"""
    import types
    from yang2024.xplane import run_yang as ry
    clock_ref = {}

    class Res:
        def __init__(self, t):
            self.timestamp, self.detected, self.confidence = t, True, 0.9
            self.raw_center_x, self.raw_center_y, self.bbox = 960.0, 505.0, (950, 495, 970, 515)

    class Detector:
        def __init__(self, *a, **k):
            self.warm = 0

        def update_altitude(self, a):
            pass

        def detect_vehicle(self, frame):
            self.warm += 1
            return True, 960.0, 505.0, (950, 495, 970, 515), 0.9

    class Shared:
        def __init__(self, *a):
            self.n = 0
            self.vision_fps = 20.0

        def get_latest(self):
            self.n += 1
            return Res(clock_ref["now"]() - 0.04), True

        @property
        def frame_count(self):
            return self.n

        def get_debug_frame(self):
            return None

    class Thread:
        def __init__(self, *a):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    fake_vd = types.SimpleNamespace(CarDetector=Detector, SharedVisionState=Shared, VisionThread=Thread)
    monkeypatch.setattr(ry, "import_senior", lambda name: fake_vd)
    orig_setup = ry.setup

    def setup_spy(*a, **k):
        out = orig_setup(*a, **k)
        clock_ref["now"] = out[1].now
        return out
    monkeypatch.setattr(ry, "setup", setup_spy)
    out = ry.main(["--dry-run", "--measure", "yolo", "--yolo-model", "x.pt", "--time", "10", "--out", str(tmp_path)])
    import json
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["summary"]
    assert s["yolo_detect_ratio"] == 1.0 and s["yolo_fps"] is not None
    assert 30 <= s["yolo_age_ms_mean"] <= 60


def test_pose_history_rejects_stale():
    from yang2024.xplane.run_yang import PoseHistory
    p = PoseHistory()
    p.add(10.0, 0.1, -0.5, 0.0, 0.0)
    assert p.at(10.1) is not None and p.at(5.0) is None


def test_overlay_draw():
    cv2 = pytest.importorskip("cv2")
    from yang2024.xplane.overlay import draw
    frame = np.zeros((1010, 1920, 3), np.uint8)
    img = draw(frame, {"mode": "Jg", "n1": 0, "R": 320.0, "R_M": 300.0, "u_cmd": -0.31, "roll": 44.0,
                       "u2": -600.0, "v2": 100.0, "u1_truth": 12.0, "v1_truth": -8.0,
                       "bbox": (950, 495, 975, 515), "conf": 0.8}, 960, 505)
    assert img.shape == (672, 1280, 3) and img.any()


def test_pick_window_position():
    from yang2024.xplane.overlay import pick_window_position
    region = {"top": 30, "left": 0, "width": 1920, "height": 1010}
    full = {"left": -2560, "top": 0, "width": 4480, "height": 1440}
    main = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    left = {"left": -2560, "top": 0, "width": 2560, "height": 1440}
    pos, _ = pick_window_position(region, [full, main, left])
    assert pos == (-2540, 40)                      # 第二螢幕在左邊
    right = {"left": 1920, "top": 200, "width": 1920, "height": 1080}
    assert pick_window_position(region, [full, main, right])[0] == (1940, 240)
    assert pick_window_position(region, [main, main])[0] is None   # 只有一個螢幕
