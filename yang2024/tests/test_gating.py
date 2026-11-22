"""量測合理性檢查與雲台轉速限制（gating.py）。"""
import math

import numpy as np
import pytest

from yang2024.xplane.gating import MeasurementGate, angle_deg, rate_limit


def los(az_deg, el_deg=-30.0):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def test_accepts_close_rejects_far_jump():
    g = MeasurementGate()
    assert g.check(los(1.0), los(0.0), lost_s=0.0)
    assert not g.check(los(25.0), los(0.0), lost_s=0.0)          # 2026-11-22 錯框一張（約 20° 以上）
    assert g.last_reason == "rejected"


def test_gate_widens_while_lost_and_reacquires_consistent_target():
    g = MeasurementGate()
    assert g.threshold(2.0) == pytest.approx(8.0 + 6.0)
    assert g.threshold(10.0) == pytest.approx(20.0)
    # 2026-11-22 右轉實驗 59.7 s：跟丟 1.5 s 時錯框偏 21.8° → 不能採用
    assert not g.check(los(21.8), los(0.0), lost_s=1.5)
    # 跟丟 1.2 s、預測偏了 50°：同一個新方向連續 6 張 → 重新抓回
    g = MeasurementGate()
    res = [g.check(los(50.0 + 0.2 * i), los(0.0), lost_s=1.2 + 0.08 * i,
                   pixel=(100.0 + 20 * i, 0.0), cam_axis=los(0.0)) for i in range(6)]
    assert res == [False] * 5 + [True] and g.last_reason == "reacquired"


def test_screen_fixed_object_is_not_reacquired():
    """固定在螢幕上的東西：相機一直轉、像素位置不變 → 永遠不採用（2026-11-22 第一次追蹤實驗）。"""
    g = MeasurementGate()
    out = []
    for i in range(10):
        cam = 5.0 * i                                             # 相機每張轉 5°
        out.append(g.check(los(60.0 + cam), los(0.0), lost_s=1.5 + 0.08 * i,
                           pixel=(-763.0, -341.0), cam_axis=los(cam)))
    assert not any(out)


def test_rate_limit():
    r = 0.0
    for _ in range(50):                                           # 1 s、命令 3 rad/s
        r = rate_limit(r, 3.0, math.radians(60), math.radians(120), 0.02)
    assert r == pytest.approx(math.radians(60))
    assert rate_limit(0.0, 3.0, math.radians(60), math.radians(120), 0.02) == pytest.approx(math.radians(120) * 0.02)
    assert rate_limit(0.0, 3.0, 2.0, 0.0, 0.02) == pytest.approx(2.0)   # acc_max = 0 → 只限轉速
    assert angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
