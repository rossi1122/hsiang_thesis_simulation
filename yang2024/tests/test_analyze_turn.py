"""analyze_turn：用合成的 log 檢查兩個指標的計算。"""
import csv
import math

import numpy as np
import pytest

from yang2024.xplane.analyze_turn import load_log, main, turn_metrics


def make_log(path, dt=0.02, T=70.0, t0=0.0, noise=0.0, seed=0):
    """吉普車 20 m/s → 40 s 起以 2.5 m/s² 減速到 7 → 48 s 起加速回 20 → 65 s 停車。
    R：60 s 前從 700 線性降到 120（45 s），再升到 320（55 s）；42–44 s 跟丟。"""
    t = np.arange(0, T, dt)
    v = np.full_like(t, 20.0)
    v[(t >= 40) & (t < 45.2)] = 20 - 2.5 * (t[(t >= 40) & (t < 45.2)] - 40)
    v[(t >= 45.2) & (t < 48)] = 7.0
    m = (t >= 48) & (t < 53.2)
    v[m] = 7 + 2.5 * (t[m] - 48)
    v[t >= 65] = 0.0
    jx = np.concatenate([[0], np.cumsum(v[:-1] * dt)])
    rng = np.random.default_rng(seed)
    jx = jx + noise * rng.standard_normal(len(t))       # X-Plane 位置抖動 → 速度雜訊
    jy = np.zeros_like(t)
    R = np.interp(t, [0, 45, 55, 70], [700, 120, 320, 320])
    valid = np.ones_like(t)
    valid[(t >= 42) & (t < 44)] = 0
    roll = np.where((t > 45) & (t < 52), 47.0, 5.0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "jx", "jy", "R_truth", "meas_valid", "roll", "pan_rate_act"])
        for row in zip(t + t0, jx, jy, R, valid, roll, np.zeros_like(t)):
            w.writerow([f"{x:.6f}" for x in row])
    return path


def test_turn_metrics_synthetic(tmp_path):
    log = load_log(make_log(tmp_path / "log.csv"))
    r, _ = turn_metrics(log, band=(150, 300), after=5.0, name="syn")
    assert r["t_brake"] == pytest.approx(40.0, abs=0.6)
    assert r["jeep_v_min"] == pytest.approx(7.0, abs=0.5)
    assert r["R_min"] == pytest.approx(120.0, abs=1.0)
    assert r["t_end"] == pytest.approx(r["t_recover"] + 5.0, abs=0.05)
    # R < 150：R 由 700→120（0–45 s）、120→320（45–55 s）；< 150 約 42.7–46.5 s
    assert r["below_band_s"] == pytest.approx(46.5 - 42.67, abs=0.2)
    # R > 300 從 54 s 起到時間窗結束
    assert r["above_band_s"] == pytest.approx(r["t_end"] - 54.0, abs=0.2)
    assert r["out_of_band_s"] == pytest.approx(r["below_band_s"] + r["above_band_s"], abs=1e-6)
    assert r["longest_loss_turn_s"] == pytest.approx(2.0, abs=0.05)
    assert r["longest_loss_after_brake_s"] == pytest.approx(2.0, abs=0.05)
    assert r["R_at_brake"] == pytest.approx(700 - 580 * 40 / 45, abs=8.0)
    assert r["max_bank_turn_deg"] == pytest.approx(47.0)
    # 停車（65 s）之後不算
    assert r["t_stop"] == pytest.approx(65.0, abs=1.2)      # 2 s 平滑 → 晚約 1 s


def test_cli_writes_outputs(tmp_path):
    p1 = make_log(tmp_path / "a.csv")
    p2 = make_log(tmp_path / "b.csv")
    out = tmp_path / "out"
    main([str(p1), str(p2), "--out", str(out)])
    assert (out / "turn_metrics.md").exists() and (out / "turn_compare.png").exists()
    text = (out / "turn_metrics.md").read_text(encoding="utf-8")
    assert "指標一" in text and "指標二" in text and "Run 2" in text


def test_wall_clock_time_and_noisy_speed(tmp_path):
    """log 的 t 是電腦時鐘、吉普車速度雜訊大（實測 20–28 m/s 跳動）時，仍要抓到真正的減速。"""
    log = load_log(make_log(tmp_path / "log.csv", t0=1795340000.0, noise=0.04, seed=3))
    r, _ = turn_metrics(log, band=(150, 300), after=5.0)
    assert r["t_brake"] == pytest.approx(40.0, abs=1.5)
    assert r["jeep_v_min"] < 9.0
    assert r["R_min"] == pytest.approx(120.0, abs=1.0)
    assert r["max_bank_turn_deg"] == pytest.approx(47.0)


def test_above_band_only_after_entering(tmp_path):
    """減速時飛機還在 300 m 外（追近中）→ 那段不算「被甩出去」；跟丟發生在出彎之後也要算進指標二。"""
    import csv as _csv
    p = make_log(tmp_path / "log.csv")
    rows = list(_csv.reader(open(p)))
    head, data = rows[0], rows[1:]
    iR, iv, it = head.index("R_truth"), head.index("meas_valid"), head.index("t")
    for r in data:
        t = float(r[it])
        r[iR] = f"{np.interp(t, [0, 42, 45, 55, 70], [700, 360, 120, 250, 250]):.4f}"   # 減速(40 s)時 R≈390
        r[iv] = "0" if 58.0 <= t < 61.0 else "1"                                        # 出彎後跟丟 3 s
    with open(p, "w", newline="") as f:
        _csv.writer(f).writerows([head] + data)
    r, _ = turn_metrics(load_log(p), band=(150, 300), after=5.0)
    assert r["R_at_brake"] > 300
    assert r["above_band_s"] == pytest.approx(0.0, abs=1e-9)
    assert r["longest_loss_turn_s"] == pytest.approx(0.0, abs=1e-9)
    assert r["longest_loss_after_brake_s"] == pytest.approx(3.0, abs=0.05)
