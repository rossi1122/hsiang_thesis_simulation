"""吉普車轉彎劇本（TurnProfile）的離線檢查，不需要 X-Plane。"""
import math

import pytest

from yang2024.jeep.jeep_turn import TurnProfile, wrap180


def drive(prof, T=80.0, dt=0.05, v0=None, tau_v=2.0):
    """簡單的車：速度一階追速度命令、航向完全跟上命令。回傳每一步的 (t, v, psi, phase)。"""
    v = prof.v_cruise if v0 is None else v0
    out = []
    for k in range(int(T / dt)):
        t = k * dt
        psi, v_cmd, phase = prof.command(t, v, dt)
        v += (v_cmd - v) * dt / tau_v
        out.append((t, v, psi, phase))
    return out


def test_curve_keeps_speed_and_turns_right_90():
    prof = TurnProfile("curve", 20.0, 200.0, turn_at=10.0, psi0=177.0)
    run = drive(prof)
    assert prof.v_turn == pytest.approx(20.0)
    assert min(v for _, v, _, _ in run) == pytest.approx(20.0, abs=1e-6)
    assert wrap180(run[-1][2] - 177.0) == pytest.approx(90.0)
    assert run[-1][3] == "exit"
    # 半徑 200 m、20 m/s → 90° 約需 π/2·200/20 = 15.7 s
    t_turn = [t for t, _, _, ph in run if ph == "turn"]
    assert t_turn[-1] - t_turn[0] == pytest.approx(math.pi / 2 * 200 / 20, abs=0.2)


def test_corner_slows_to_lateral_limit_then_recovers():
    prof = TurnProfile("corner", 20.0, 20.0, turn_at=10.0, psi0=177.0, a_lat=3.0)
    run = drive(prof)
    assert prof.v_turn == pytest.approx(math.sqrt(3.0 * 20.0))          # ≈ 7.7 m/s
    phases = [ph for _, _, _, ph in run]
    assert phases.index("brake") < phases.index("turn") < phases.index("exit")
    v_in_turn = [v for _, v, _, ph in run if ph == "turn"]
    assert max(v_in_turn) <= prof.v_turn + 1.0 + 1e-6                   # 彎中側向加速度 ≤ 約 0.3 g
    assert run[-1][1] > 19.0                                            # 出彎後加速回原速
    assert wrap180(run[-1][2] - 177.0) == pytest.approx(90.0)


def test_noslow_and_left_turn():
    prof = TurnProfile("noslow", 20.0, 20.0, turn_at=5.0, psi0=10.0, direction="left")
    run = drive(prof, T=30.0)
    assert prof.v_turn == pytest.approx(20.0)
    assert wrap180(run[-1][2] - 10.0) == pytest.approx(-90.0)


def test_bad_mode():
    with pytest.raises(ValueError):
        TurnProfile("drift", 20.0, 20.0, 5.0, 0.0)


def test_turn_by_distance_corner_starts_braking_early():
    prof = TurnProfile("corner", 20.0, 20.0, turn_at=999.0, psi0=356.0, a_lat=3.0, a_brake=2.5, turn_dist=500.0)
    v, dist, dt = 20.0, 0.0, 0.05
    start_turn = None
    for k in range(int(60 / dt)):
        _, v_cmd, phase = prof.command(k * dt, v, dt, dist)
        v += (v_cmd - v) * dt / 0.3
        dist += v * dt
        if phase == "turn" and start_turn is None:
            start_turn = dist
    assert prof.brake_lead == pytest.approx((400 - 60) / 5.0)
    assert start_turn == pytest.approx(500.0, abs=25.0)      # 轉彎開始的位置接近指定距離


def test_command_never_runs_far_ahead_and_waits_for_actual_turn():
    """X-Plane 實測：轉向打滿時車只轉 9°/s，命令不能一路跑到 90°、也不能沒轉完就加速。"""
    prof = TurnProfile("corner", 20.0, 20.0, turn_at=0.0, psi0=0.0, a_lat=3.0)
    dt, actual, v = 0.05, 0.0, 20.0
    for k in range(int(40 / dt)):
        _, v_cmd, phase = prof.command(k * dt, v, dt, None, actual)
        v += (v_cmd - v) * dt / 0.5
        if phase in ("turn", "exit"):
            actual = min(90.0, actual + 9.0 * dt)          # 車最多 9°/s
        assert prof.turned <= actual + prof.lead_max + 1e-9
        if phase == "turn":
            assert v_cmd == pytest.approx(prof.v_turn)
    assert phase == "exit" and actual >= 85.0
