"""run_yang / step_test / camera_check 共用：建立連線、紀錄、存檔。"""
import csv
import json
import math
from datetime import datetime

import numpy as np

from .config import PROFILES, build, log_dir
from .fake_xplane import FakeXPlane, RealClock
from .xp_link import XPLink


def add_profile_arg(ap):
    ap.add_argument("--profile", choices=PROFILES, default="scaled",
                    help="scaled = 尺度版 V 30 m/s、H 300 m（預設）；original = 論文原值 V 16 m/s、H 100 m")


def setup(dry_run=False, jeep_speed=None, profile="scaled", vc=None, **scale_overrides):
    """回傳 (link, clock, params, xp, k)。dry_run 用假的 X-Plane。vc：論文尺度的目標速度（覆寫 jeep_speed_paper）。"""
    params, xp, k = build(profile=profile, **scale_overrides)
    if vc is not None:
        xp["scenario"]["jeep_speed_paper"] = float(vc)
    print(f"[版本] {profile}：V = {params['V_t']:g} m/s、H = {params['H']:g} m、"
          f"R_M = {params['R_M']:g} m、u_max = {params['u_max']:.4f} rad/s、"
          f"吉普車應為 {jeep_speed_target(xp, k):.1f} m/s")
    if dry_run:
        speed = jeep_speed if jeep_speed is not None else jeep_speed_target(xp, k)
        fake = FakeXPlane(jeep_speed=speed)
        return XPLink(xp, client=fake, clock=fake.now), fake, params, xp, k
    clock = RealClock()
    return XPLink(xp, clock=clock.now), clock, params, xp, k


def jeep_speed_target(xp, k):
    """吉普車應該開的速度 = 論文 V_c × k_V（scaled 7 → 13.1 m/s；original 7 → 7 m/s）。"""
    return xp["scenario"]["jeep_speed_paper"] * k["k_V"]


def wait_for_jeep(link, clock, timeout=10.0):
    t0 = clock.now()
    while clock.now() - t0 < timeout:
        st = link.read()
        if st.jeep_ok:
            return st
        clock.sleep(0.2)
    raise RuntimeError("讀不到吉普車位置：確認筆電的 X-Plane 有開、多人連線有設定、"
                       "飛機 PC 的「設定 → 網路 → 接收數據」看得到 192.168.50.39")


class Recorder:
    def __init__(self):
        self.rows = []

    def add(self, **row):
        self.rows.append(row)

    def column(self, key):
        return np.array([r.get(key, np.nan) for r in self.rows], dtype=float)

    def save_csv(self, path):
        if not self.rows:
            return
        keys = list(self.rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def output_dir(xp, name):
    d = log_dir(xp) / f"{datetime.now():%Y%m%d_%H%M%S}_{xp.get('profile', 'scaled')}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not math.isfinite(float(o)) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean(obj), fh, ensure_ascii=False, indent=2)
