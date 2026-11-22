"""吉普車右轉 90° 的測試與情境腳本（在 jeep 筆電的 X-Plane 執行）。

沿用學長的 xplane_sim_jeep/jeep_controller.py（航向 PID、速度 PI、home frame 座標），原檔不修改；
這裡只換掉「航向、速度命令怎麼隨時間變化」。

三種模式（--mode）：
  curve  ：大半徑曲線，速度不變（老師說「除非是曲線」的情況）
  corner ：一般路口轉彎，像真實駕駛：彎中速度受側向加速度限制 v ≤ √(a_lat·r)，
           所以會在進彎時減速、出彎後加速（「轉彎會掉速」是物理造成的，不是任意設定）
  noslow ：測試用：小半徑但不減速，看 X-Plane 的吉普車會怎樣（打滑、翻車，或轉不了那麼急）

用法（repo 根目錄）：
  python yang2024/jeep/jeep_turn.py --mode curve  --speed 20 --radius 200
  python yang2024/jeep/jeep_turn.py --mode corner --speed 20 --radius 20
  python yang2024/jeep/jeep_turn.py --mode noslow --speed 20 --radius 20
log 存在執行目錄的 jeep_turn_logs/（含 wall_time，可和飛機 log 對時）。
"""
import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

G = 9.81
MODES = ("curve", "corner", "noslow")
DEFAULT_RADIUS = {"curve": 200.0, "corner": 20.0, "noslow": 20.0}


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class TurnProfile:
    """依時間與實際狀態產生 (航向命令, 速度命令, 階段)。不依賴 X-Plane，可單獨測試。

    階段：straight → (brake，只有 corner) → turn → exit
    - turn：航向命令以 ω = v / r 前進（用實際速度，讓轉彎半徑接近 r），轉滿 turn_deg 為止
    - corner 的彎中速度 v_turn = min(v_cruise, √(a_lat·r))；進彎前速度命令以 a_brake 往下降
    """

    def __init__(self, mode, v_cruise, radius, turn_at, psi0, a_lat=3.0, a_brake=2.5,
                 turn_deg=90.0, direction="right", brake_timeout=20.0):
        if mode not in MODES:
            raise ValueError(f"mode 必須是 {MODES}")
        self.mode, self.v_cruise, self.radius = mode, float(v_cruise), float(radius)
        self.turn_at, self.psi0, self.turn_deg = float(turn_at), float(psi0), float(turn_deg)
        self.a_brake, self.brake_timeout = float(a_brake), float(brake_timeout)
        self.sign = 1.0 if direction == "right" else -1.0      # 真北順時針為正 → 右轉是 +
        self.v_turn = (min(self.v_cruise, math.sqrt(a_lat * self.radius)) if mode == "corner"
                       else self.v_cruise)
        self.phase = "straight"
        self.v_cmd = self.v_cruise
        self.turned = 0.0           # 航向命令已經轉了幾度
        self.t_phase = 0.0

    def psi_cmd(self):
        return (self.psi0 + self.sign * self.turned) % 360.0

    def command(self, t, speed, dt):
        if self.phase == "straight" and t >= self.turn_at:
            self.phase = "brake" if self.v_turn < self.v_cruise - 0.5 else "turn"
            self.t_phase = t
        if self.phase == "brake":
            self.v_cmd = max(self.v_turn, self.v_cmd - self.a_brake * dt)
            slowed = self.v_cmd <= self.v_turn and speed <= self.v_turn + 1.0
            if slowed or t - self.t_phase > self.brake_timeout:
                self.phase, self.t_phase = "turn", t
        if self.phase == "turn":
            self.v_cmd = self.v_turn
            rate = math.degrees(max(speed, 1.0) / self.radius)
            self.turned = min(self.turn_deg, self.turned + rate * dt)
            if self.turned >= self.turn_deg:
                self.phase, self.t_phase = "exit", t
        if self.phase == "exit":
            self.v_cmd = self.v_cruise
        return self.psi_cmd(), self.v_cmd, self.phase


def import_jeep_controller():
    """學長的 JeepController（同時支援：在 repo 根目錄執行，或把本檔複製到 xplane_sim_jeep/ 執行）。"""
    here = Path(__file__).resolve().parent
    for p in (here, here.parents[1] / "xplane_sim_jeep"):
        if (p / "jeep_controller.py").exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from jeep_controller import JeepController
    return JeepController


def main():
    ap = argparse.ArgumentParser(description="吉普車右轉 90° 測試（curve / corner / noslow）")
    ap.add_argument("--mode", choices=MODES, default="corner")
    ap.add_argument("--speed", type=float, default=20.0, help="直線速度 m/s")
    ap.add_argument("--radius", type=float, default=None, help="轉彎半徑 m（預設 curve 200、其他 20）")
    ap.add_argument("--turn-at", type=float, default=30.0, help="開始轉彎（或開始減速）的時間 s")
    ap.add_argument("--a-lat", type=float, default=3.0, help="corner：彎中側向加速度上限 m/s²（3 ≈ 0.3 g）")
    ap.add_argument("--a-brake", type=float, default=2.5, help="corner：進彎減速度 m/s²")
    ap.add_argument("--direction", choices=("right", "left"), default="right")
    ap.add_argument("--heading", type=float, default=None,
                    help="直線段航向（度，真北順時針）；不給 = 用吉普車擺好時的車頭方向（換跑道、停機坪不用改）")
    ap.add_argument("--max-steer", type=float, default=0.15, help="轉向命令上限（學長預設 0.15）")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--roll-abort", type=float, default=35.0, help="車身滾轉超過幾度就停車（翻車保護）")
    ap.add_argument("--hz", type=int, default=20)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=49009)
    args = ap.parse_args()
    radius = args.radius if args.radius is not None else DEFAULT_RADIUS[args.mode]

    JeepController = import_jeep_controller()
    ctrl = JeepController(target_speed=args.speed, target_heading=args.heading or 0.0, control_hz=args.hz,
                          xp_host=args.host, xp_port=args.port)
    ctrl.max_steer = args.max_steer
    ctrl.initialize()
    psi0 = args.heading
    if psi0 is None:
        st0 = None
        for _ in range(20):
            st0 = ctrl.get_state()
            if st0 is not None:
                break
            time.sleep(0.1)
        if st0 is None:
            print("[錯誤] 讀不到吉普車狀態，請確認 X-Plane 已開啟、XPlaneConnect 正常")
            return
        psi0 = st0["heading"]
        print(f"[航向] 沒有給 --heading，用目前車頭方向 {psi0:.1f}° 當直線段航向")

    prof = TurnProfile(args.mode, args.speed, radius, args.turn_at, psi0, a_lat=args.a_lat,
                       a_brake=args.a_brake, direction=args.direction)
    print(f"[劇本] {args.mode}：直線 {args.speed:.1f} m/s、{args.turn_at:.0f} s 後{'右' if args.direction == 'right' else '左'}轉 90°、"
          f"半徑 {radius:.0f} m、彎中速度 {prof.v_turn:.1f} m/s"
          f"（側向加速度 {prof.v_turn ** 2 / radius / G:.2f} g）")

    os.makedirs("jeep_turn_logs", exist_ok=True)
    path = os.path.join("jeep_turn_logs", f"jeep_turn_{datetime.now():%Y%m%d_%H%M%S}_{args.mode}.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["wall_time", "time", "phase", "psi_cmd", "v_cmd", "heading", "groundspeed", "dist_m", "turned_deg",
                "roll_deg", "yaw_rate_dps", "a_lat_mps2", "x_position", "y_position", "local_x", "local_z",
                "lat", "lon", "heading_err", "steer", "throttle", "brake"])
    print(f"[CSV] {path}")

    dt = 1.0 / args.hz
    t0 = time.time()
    prev_heading = None
    prev_xz = None
    dist = 0.0          # 累積行駛距離（m），可用來量跑道／停機坪長度
    actual_turn = 0.0
    stats = dict(v_turn_start=None, v_min=None, roll_max=0.0, alat_max=0.0, t_turn_start=None,
                 t_turn_end=None, radius=[])
    aborted = ""
    try:
        while True:
            loop = time.time()
            t = loop - t0
            if t > args.duration:
                break
            st = ctrl.get_state()
            if st is None:
                time.sleep(dt)
                continue
            try:
                roll = ctrl.client.getDREF("sim/flightmodel/position/phi")[0]
                yaw_rate = ctrl.client.getDREF("sim/flightmodel/position/R")[0]   # deg/s
            except Exception:
                roll, yaw_rate = float("nan"), float("nan")
            heading, speed = st["heading"], st["groundspeed"]
            if prev_heading is not None:
                actual_turn += wrap180(heading - prev_heading)
            prev_heading = heading
            if prev_xz is not None:
                dist += math.hypot(st["local_x"] - prev_xz[0], st["local_z"] - prev_xz[1])
            prev_xz = (st["local_x"], st["local_z"])

            psi_cmd, v_cmd, phase = prof.command(t, speed, dt)
            h_err = ctrl.heading_error(heading, psi_cmd)
            steer = ctrl.compute_rudder(h_err)
            throttle, brake = ctrl.compute_throttle_and_brake(v_cmd - speed)
            ctrl.client.sendCTRL([0, 0, 0, throttle])
            ctrl.client.sendDREF("sim/joystick/yoke_roll_ratio", steer)
            ctrl.client.sendDREF("sim/cockpit2/controls/left_brake_ratio", brake)
            ctrl.client.sendDREF("sim/cockpit2/controls/right_brake_ratio", brake)

            a_lat = speed * math.radians(yaw_rate) if math.isfinite(yaw_rate) else float("nan")
            if phase in ("brake", "turn", "exit"):
                stats["v_min"] = speed if stats["v_min"] is None else min(stats["v_min"], speed)
            if phase == "turn" and stats["t_turn_start"] is None:
                stats["t_turn_start"], stats["v_turn_start"] = t, speed
            if stats["t_turn_start"] is not None and stats["t_turn_end"] is None \
                    and abs(actual_turn) >= prof.turn_deg - 5.0:
                stats["t_turn_end"] = t
            if math.isfinite(roll):
                stats["roll_max"] = max(stats["roll_max"], abs(roll))
            if math.isfinite(a_lat):
                stats["alat_max"] = max(stats["alat_max"], abs(a_lat))
                if phase == "turn" and abs(math.radians(yaw_rate)) > 0.02:
                    stats["radius"].append(speed / abs(math.radians(yaw_rate)))

            w.writerow([f"{loop:.6f}", f"{t:.3f}", phase, f"{psi_cmd:.3f}", f"{v_cmd:.3f}", f"{heading:.3f}",
                        f"{speed:.3f}", f"{dist:.1f}", f"{actual_turn:.2f}", f"{roll:.3f}", f"{yaw_rate:.3f}", f"{a_lat:.3f}",
                        f"{st['x_position']:.3f}", f"{st['y_position']:.3f}", f"{st['local_x']:.3f}",
                        f"{st['local_z']:.3f}", f"{st['lat']:.8f}", f"{st['lon']:.8f}", f"{h_err:.3f}",
                        f"{steer:.4f}", f"{throttle:.4f}", f"{brake:.4f}"])
            if int(t / dt) % args.hz == 0:
                print(f"{t:6.1f} s  {phase:8s}  v_cmd {v_cmd:5.1f}  v {speed:5.1f} m/s  已開 {dist:6.0f} m  "
                      f"航向命令 {psi_cmd:6.1f}  航向 {heading:6.1f}  已轉 {actual_turn:+6.1f}°  "
                      f"滾轉 {roll:+5.1f}°  側向 {a_lat / G:+.2f} g")
            if math.isfinite(roll) and abs(roll) > args.roll_abort:
                aborted = f"車身滾轉 {roll:.0f}° 超過 {args.roll_abort:.0f}°（可能翻車），停車"
                break
            time.sleep(max(0.0, dt - (time.time() - loop)))
    except KeyboardInterrupt:
        aborted = "Ctrl+C"
    finally:
        ctrl.stop()
        f.close()

    r = sorted(stats["radius"])
    print("\n" + "=" * 60)
    print(f"  模式 {args.mode}，命令半徑 {radius:.0f} m，彎中速度命令 {prof.v_turn:.1f} m/s")
    if aborted:
        print(f"  ⚠️ 中止：{aborted}")
    if stats["t_turn_start"] is None:
        print("  還沒開始轉彎就結束了（--duration 要比 --turn-at 長）")
    else:
        dur = (stats["t_turn_end"] - stats["t_turn_start"]) if stats["t_turn_end"] else float("nan")
        print(f"  開始轉彎時速度 {stats['v_turn_start']:.1f} m/s；轉彎期間最低速度 {stats['v_min']:.1f} m/s")
        print(f"  實際轉了 {actual_turn:+.0f}°；轉滿 {prof.turn_deg - 5:.0f}° 花 {dur:.1f} s")
        if r:
            print(f"  實際轉彎半徑（中位數）{r[len(r) // 2]:.0f} m")
        print(f"  最大側向加速度 {stats['alat_max'] / G:.2f} g；最大車身滾轉 {stats['roll_max']:.1f}°")
    print(f"  總行駛距離 {dist:.0f} m")
    print(f"  log：{path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
