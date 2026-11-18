"""Yang 2024 在 X-Plane 11 的閉迴路（Stage 3 真值特徵 / Stage 4 YOLO）。

每個 MPC 週期（Δt，縮放後 0.08 s）—— 對應論文 Algorithm 1：
  1. 量測目標特徵 (u1, v1)：
       truth：用吉普車真值投影（理想偵測，先驗證控制本身）
       yolo ：學長的 YOLO 執行緒；用影像擷取當下的姿態與雲台角度做轉換
  2. 轉成 Ideal State (u2, v2)、垂直相機 (u3, v3)、g1（Eq. 7、15、20）
  3. 切換式 MPC / P1,soft / LSM 算出 u_psi（和數值模擬同一份程式）
  4. 雲台律（Eq. 9）算出雲台角速度
每一圈（control_hz，預設 50 Hz）：
  傾斜轉彎內迴路把 u_psi 變成副翼、方向舵、升降舵、油門；雲台角度依真實經過時間積分後送出。

兩種設定（2026-11-18）：
  改良版（預設）：SIYI A8 mini 跟隨模式的三軸增穩雲台（xp_params.yaml 的 gimbal）+ 目標跟丟處理（lost_target）：
      - 雲台在「世界座標」追視線方向：量到目標時用 Eq. 9 的比例律（λ = 0.5）把視線拉過去，飛機轉彎自動抵消；
      - 偵測不到時用 Eq. 10（Yang 自己的預測模型）+ 實測航向變化率推算 s2，雲台對準推算的方向，MPC 用推算的 s2；
        只用影像與自身運動，深度只當增益（H / sin），不做目標定位。
  --yang-original：Yang 2024 原版（兩軸機身雲台、Eq. 9 像素律、偵測不到時沿用舊值）。A–E2 與 YOLO 版 A 都是原版。

用法（飛機 PC，repo 根目錄，dive_sim 環境）：
    python -m yang2024.xplane.run_yang                       # 切換式 MPC、真值特徵、論文起始位置
    python -m yang2024.xplane.run_yang --controller lsm
    python -m yang2024.xplane.run_yang --start far           # 從吉普車後方 900 m 起始
    python -m yang2024.xplane.run_yang --measure yolo --yolo-model <best.pt 路徑>
    python -m yang2024.xplane.run_yang --dry-run --time 30   # 不開 X-Plane，用假的 X-Plane 測試程式
    python -m yang2024.xplane.run_yang --yang-original       # Yang 2024 原版（重現用）
"""
import argparse
import math
import sys
from collections import deque
from pathlib import Path

# Windows：YOLO 模式要先載入 torch，再載入 casadi（mpc.py）。
# 2026-11-17：casadi 先載入時，torch 會出現 WinError 127「找不到指定的程序 … fbgemm.dll」
# （camera_check 沒有 casadi，所以同一個 YOLO 在那裡正常）。
if "yolo" in sys.argv:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — 讓後面 start_yolo 再報完整錯誤
        print(f"[YOLO] 預先載入 torch 失敗：{exc}")

import numpy as np

from .. import geometry as geo
from ..baselines import lsm_u_psi
from ..gimbal_model import GimbalModel
from ..mpc import Switcher, YangMPC
from .autopilot import BankToTurnAutopilot
from .camera import XPCamera
from .common import add_profile_arg, jeep_speed_target, Recorder, output_dir, save_json, setup, wait_for_jeep
from .xp_link import import_senior, place_relative_to_jeep


class PoseHistory:
    """YOLO 結果有延遲：用影像擷取當下的相機姿態做 Ideal State 轉換（避免把自身運動誤當成目標運動）。

    每一筆存 (時間, C_H = 相機列在水平航向座標, ψ)。"""

    def __init__(self, maxlen=200):
        self.buf = deque(maxlen=maxlen)

    def add(self, t, C_H, psi):
        self.buf.append((t, C_H, psi))

    def at(self, t, max_gap=0.3):
        if not self.buf:
            return None
        best = min(self.buf, key=lambda r: abs(r[0] - t))
        return best if abs(best[0] - t) <= max_gap else None   # 例如暫停中拍的影像：沒有對應的姿態


def start_yolo(model_path, cam, altitude):
    """載入 YOLO、先空跑一次（GPU 第一次推論要 1–3 s），再啟動截圖執行緒。

    2026-11-17 YOLO 版實驗 A：原本在飛機解除暫停前一刻才啟動，暖機期間飛機已在轉彎，
    目標漂出畫面，前 7.4 s 都偵測不到。現在在擺放飛機之前就啟動並暖機。
    """
    import time as _time
    vd = import_senior("vision_driver")
    detector = vd.CarDetector(model_path, use_tracker=False)
    detector.update_altitude(altitude)
    t = _time.time()
    detector.detect_vehicle(np.zeros((cam.height, cam.width, 3), np.uint8))
    print(f"[YOLO] 暖機完成（第一次推論 {(_time.time() - t) * 1e3:.0f} ms）")
    shared = vd.SharedVisionState(cam.width, cam.height)
    thread = vd.VisionThread(detector, shared, cam.width, cam.height)
    thread.start()
    return detector, shared, thread


def wait_first_detection(vision, clock, since, timeout=8.0):
    """擺好飛機、雲台對準後（模擬暫停中），等 YOLO 用之後拍的畫面偵測到目標。"""
    _, shared, _ = vision
    t0 = clock.now()
    while clock.now() - t0 < timeout:
        res, _ = shared.get_latest()
        if res.detected and res.timestamp > since:
            print(f"[YOLO] 已偵測到吉普車（信心 {res.confidence:.2f}），開始")
            return True
        clock.sleep(0.05)
    print("[YOLO] ⚠️ 8 秒內沒有偵測到吉普車：確認 X-Plane 在主螢幕、3D 座艙視角、沒有被視窗擋住。仍然開始。")
    return False


def summarize(rec, params, k, dt_mpc):
    t = rec.column("t")
    mpc = rec.column("mpc_tick") > 0
    R = rec.column("R_truth")
    dts = np.diff(t, prepend=t[0])
    out = R > params["R_M"]
    inside = np.where(~out & np.isfinite(R))[0]
    s2 = np.hypot(rec.column("u2"), rec.column("v2"))[mpc]
    u = rec.column("u_cmd")[mpc]
    solve = rec.column("solve_ms")[mpc]
    return {
        "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
        "loop_hz": float(len(t) / max(t[-1] - t[0], 1e-6)) if len(t) > 1 else 0.0,
        "mpc_hz": float(mpc.sum() / max(t[-1] - t[0], 1e-6)) if len(t) > 1 else 0.0,
        "rmse_s2_px": float(np.sqrt(np.nanmean(s2**2))),
        "rmse_s2_norm_x537": float(np.sqrt(np.nanmean(s2**2)) * 537.0 / params["f"]),
        "R_max_m": float(np.nanmax(R)) if np.isfinite(R).any() else None,
        "R_RM_ratio_max": float(np.nanmax(R) / params["R_M"]) if np.isfinite(R).any() else None,
        "time_outside_RM_s": float(np.sum(dts[out])),
        "t_enter_RM_s": float(t[inside[0]] - t[0]) if inside.size else None,
        "u_sat_ratio": float(np.mean(np.abs(u) >= 0.999 * params["u_max"])),
        "solve_ms_mean": float(np.nanmean(solve)) if np.isfinite(solve).any() else None,
        "solve_ms_max": float(np.nanmax(solve)) if np.isfinite(solve).any() else None,
        "psi_rate_tracking_rms": float(np.sqrt(np.nanmean((rec.column("psi_rate_coord") - rec.column("u_cmd"))**2))),
        "elevator_std": float(np.nanstd(rec.column("elevator"))),
        "agl_rms_err_m": float(np.sqrt(np.nanmean((rec.column("agl") - params["H"])**2))),
        "agl_max_err_m": float(np.nanmax(np.abs(rec.column("agl") - params["H"]))),
        "airspeed_rms_err": float(np.sqrt(np.nanmean((rec.column("airspeed") - params["V_t"])**2))),
        "roll_max_deg": float(np.nanmax(np.abs(rec.column("roll")))),
        "yolo_detect_ratio": float(np.nanmean(rec.column("detected")[mpc])),
        **yolo_stats(rec, mpc),
        **tracking_stats(rec, mpc),
        "scale": k,
    }


def tracking_stats(rec, mpc):
    """量測有效比例、推算比例、最長連續跟丟、雲台關節碰到極限比例、最大影像傾斜。"""
    t = rec.column("t")
    valid = rec.column("meas_valid")[mpc]
    pred = rec.column("predicted")[mpc]
    lost = rec.column("lost_s")
    sat = rec.column("gimbal_sat")
    cam_roll = rec.column("cam_roll")
    episodes = 0
    if lost.size:
        was = False
        for x in lost:
            now_lost = x > 0.5
            if now_lost and not was:
                episodes += 1
            was = now_lost
    return {
        "meas_valid_ratio": float(np.nanmean(valid)) if valid.size else None,
        "predicted_ratio": float(np.nanmean(pred)) if pred.size else None,
        "lost_max_s": float(np.nanmax(lost)) if lost.size else None,
        "lost_episodes_over_0p5s": int(episodes),
        "gimbal_sat_ratio": float(np.nanmean(sat)) if sat.size else None,
        "cam_roll_max_deg": float(np.nanmax(np.abs(cam_roll))) if cam_roll.size else None,
    }


def yolo_stats(rec, mpc):
    """YOLO 模式：偵測結果的延遲、更新頻率、和真值投影的像素誤差（truth 模式全部是 None）。"""
    ts = rec.column("yolo_ts")[mpc]
    t = rec.column("t")[mpc]
    ok = np.isfinite(ts)
    if not ok.any():
        return {"yolo_age_ms_mean": None, "yolo_hz": None, "yolo_err_px_rms": None, "yolo_err_px_max": None}
    age = (t[ok] - ts[ok]) * 1e3
    uniq = np.unique(ts[ok])
    det = (rec.column("detected")[mpc] > 0) & ok
    err = np.hypot(rec.column("u1")[mpc] - rec.column("u1_truth")[mpc],
                   rec.column("v1")[mpc] - rec.column("v1_truth")[mpc])[det]
    err = err[np.isfinite(err)]
    return {
        "yolo_age_ms_mean": float(np.mean(age)),
        "yolo_hz": float((len(uniq) - 1) / max(uniq[-1] - uniq[0], 1e-6)) if len(uniq) > 1 else None,
        "yolo_err_px_rms": float(np.sqrt(np.mean(err**2))) if err.size else None,   # 量測（控制當下）和真值投影的差
        "yolo_err_px_max": float(np.max(err)) if err.size else None,
    }


def plot(rec, params, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = rec.column("t")
    t = t - t[0]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    a = ax[0, 0]
    a.plot(rec.column("x"), rec.column("y"), label="UAV")
    a.plot(rec.column("jx"), rec.column("jy"), "k", lw=2, label="jeep")
    a.set_aspect("equal", adjustable="datalim")
    a.set(title="trajectory", xlabel="x east (m)", ylabel="y north (m)")
    a = ax[0, 1]
    a.plot(t, rec.column("R_truth"), label="R (jeep truth)")
    a.axhline(params["R_M"], color="k", ls="--")
    a.set(title="horizontal distance R (m)", xlabel="t (s)")
    a = ax[0, 2]
    a.plot(t, rec.column("u_cmd"), label="u_psi command")
    a.plot(t, rec.column("psi_rate_coord"), label="actual heading rate (g·tanφ/V)")
    a.axhline(params["u_max"], color="k", ls="--", lw=1)
    a.axhline(-params["u_max"], color="k", ls="--", lw=1)
    a.set(title="u_psi (rad/s)", xlabel="t (s)")
    a = ax[1, 0]
    a.plot(t, rec.column("phi_cmd"), label="roll command")
    a.plot(t, rec.column("roll"), label="roll")
    a.set(title="roll (deg)", xlabel="t (s)")
    a = ax[1, 1]
    a.plot(t, rec.column("agl") - params["H"], label="AGL − H (m)")
    a.plot(t, rec.column("airspeed") - params["V_t"], label="airspeed − V (m/s)")
    a.set(title="altitude / airspeed error", xlabel="t (s)")
    a = ax[1, 2]
    a.plot(t, rec.column("u2"), label="u2")
    a.plot(t, rec.column("v2"), label="v2")
    a.plot(t, rec.column("u1"), label="u1", alpha=0.6)
    a.plot(t, rec.column("v1"), label="v1", alpha=0.6)
    a.set(title="image features (px)", xlabel="t (s)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def start_distance(xp, k, params):
    """這個起始情境應有的水平距離：paper = 論文 (−85, 0)×k_L；far = far_distance_paper×k_L。"""
    if xp["scenario"]["start"] == "far":
        return float(xp["scenario"]["far_distance_paper"] * k["k_L"])
    return float(np.hypot(*params["uav_init"]))


def run_checks(rec, params, v_jeep, R0_target):
    """飛行中（沒暫停）的吉普車實際速度、起始水平距離，並在速度不對時提醒。"""
    t, jx, jy, R = (rec.column(c) for c in ("t", "jx", "jy", "R_truth"))
    ok = np.isfinite(jx) & np.isfinite(jy)
    v_run = float("nan")
    if ok.sum() > 10 and t[ok][-1] - t[ok][0] > 1.0:
        v_run = float(math.hypot(jx[ok][-1] - jx[ok][0], jy[ok][-1] - jy[ok][0]) / (t[ok][-1] - t[ok][0]))
        if abs(v_run - v_jeep) > 0.1 * v_jeep:
            print(f"[檢查] ⚠️ 飛行中吉普車平均 {v_run:.1f} m/s，和應有的 {v_jeep:.1f} m/s 差超過 10%：這次結果不能和論文比")
    R0 = float(R[np.isfinite(R)][0]) if np.isfinite(R).any() else float("nan")
    print(f"[檢查] 吉普車飛行中平均 {v_run:.2f} m/s（應為 {v_jeep:.2f}）；起始水平距離 {R0:.0f} m（應為 {R0_target:.0f} m）")
    return {"jeep_speed_run": v_run, "R_start_m": R0, "R_start_target_m": R0_target}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=("switch", "p1soft", "lsm"), default="switch")
    ap.add_argument("--measure", choices=("truth", "yolo"), default="truth")
    ap.add_argument("--yolo-model", default=None)
    ap.add_argument("--start", choices=("paper", "far"), default=None, help="覆寫 xp_params.yaml 的 scenario.start")
    ap.add_argument("--time", type=float, default=None, help="模擬秒數（預設 xp_params.yaml 的 max_time_s）")
    ap.add_argument("--no-reset", action="store_true", help="不重新擺放飛機，從目前位置開始")
    ap.add_argument("--dry-run", action="store_true", help="用假的 X-Plane（不需要 X-Plane）")
    ap.add_argument("--out", default=None, help="輸出資料夾（預設 results/xplane/<時間>_yang_...）")
    ap.add_argument("--show", action="store_true",
                    help="開即時顯示視窗（模式、R、YOLO 框、真值十字）。視窗要放在另一個螢幕，見 --show-x")
    ap.add_argument("--record", action="store_true",
                    help="（現在預設就會錄影，保留這個選項只為了相容舊指令）")
    ap.add_argument("--no-record", action="store_true",
                    help="這次不要錄影（預設每次都把顯示畫面錄成結果資料夾裡的 overlay.mp4）")
    ap.add_argument("--show-x", type=int, default=None,
                    help="顯示視窗左上角 x（預設自動：放在 X-Plane 以外的另一個螢幕）")
    ap.add_argument("--show-y", type=int, default=None)
    ap.add_argument("--yang-original", action="store_true",
                    help="Yang 2024 原版（兩軸機身雲台、Eq. 9 像素律、偵測不到時沿用舊值）。預設是改良版（增穩雲台 + 跟丟處理）")
    ap.add_argument("--vc", type=float, default=None,
                    help="論文尺度的目標速度 V_c（論文用 5、7、9；預設 7）。吉普車實際速度 = V_c × k_V，例如 9 → 16.9 m/s")
    add_profile_arg(ap)
    args = ap.parse_args(argv)

    link, clock, params, xp, k = setup(dry_run=args.dry_run, profile=args.profile, vc=args.vc)
    if args.start:
        xp["scenario"]["start"] = args.start
    T = args.time if args.time is not None else xp["loop"]["max_time_s"]
    if args.measure == "yolo" and not args.yolo_model:
        raise SystemExit("--measure yolo 需要 --yolo-model（例如 yolo_model/yolo_jeep12/weights/best.pt）")

    f, H, V = params["f"], params["H"], params["V_t"]
    alpha = geo.alpha_from(H, params["R_d"])
    dt_mpc = params["dt"]
    cam = XPCamera(xp["camera"])
    autopilot = BankToTurnAutopilot(xp["autopilot"], V_cmd=V, H_cmd=H)
    mpc = YangMPC(params) if args.controller != "lsm" else None
    lam_u, lam_v = params["gimbal_lambda"]
    tilt_min, tilt_max = params["gimbal_tilt_range"]
    rate_max = params["gimbal_rate_max"]

    # ---------------- 原版 / 改良版
    enhanced = not args.yang_original
    gimbal = GimbalModel.from_config(xp.get("gimbal", {}), params, mode=None if enhanced else "body")
    follow = gimbal.mode == "follow"
    lost_cfg = xp.get("lost_target", {})
    lost_handling = enhanced and lost_cfg.get("enabled", True)
    fov_check = enhanced and lost_cfg.get("truth_fov_check", True)
    print(f"[設定] {'改良版' if enhanced else 'Yang 2024 原版'}：雲台 {gimbal.mode}"
          f"{'（SIYI A8 mini 跟隨模式，三軸增穩）' if follow else '（兩軸、相對機身）'}、"
          f"跟丟處理 {'開' if lost_handling else '關'}")

    print(f"[Yang X-Plane] V = {V} m/s、H = {H} m、R_d = {params['R_d']:.0f} m、R_M = {params['R_M']:.0f} m、"
          f"u_max = {params['u_max']:.4f} rad/s、Δt = {dt_mpc:.3f} s、f = {f:.1f} px")
    v_jeep = jeep_speed_target(xp, k)
    print(f"[Yang X-Plane] 吉普車速度應為 {xp['scenario']['jeep_speed_paper']} × {k['k_V']:.3f} = "
          f"{v_jeep:.1f} m/s（jeep_controller.py --speed）")

    def depth_of(v2):
        if params["depth_model"] == "const":
            return params["depth_const"]
        return geo.ideal_depth(v2, alpha, f, H)

    # ---------------- 雲台狀態
    # body  ：g = (pan, tilt)，相對機身（原版）
    # follow：g = (視線方位 az_w, 仰角 el)，在世界座標；送出時 pan = az_w − ψ（跟隨模式：相對機頭的水平方向）
    g = [0.0, 0.0]
    C_body_cmd = geo.camera_rows(0.0, 0.0)      # 目前送給 X-Plane 的相機姿態（相對機身）

    def command_gimbal(st_now):
        """把雲台狀態換成相機姿態、送給 X-Plane。回傳 (指令 pan, 指令 tilt, info)。"""
        nonlocal C_body_cmd
        pitch_r, roll_r = math.radians(st_now.pitch_deg), math.radians(st_now.roll_deg)
        if follow:
            pan_c, tilt_c = geo.wrap(g[0] - st_now.psi), g[1]
        else:
            pan_c, tilt_c = g[0], g[1]
        _, C_body_cmd, info = gimbal.rows(pan_c, tilt_c, pitch_r, roll_r)
        if follow:
            link.gimbal_head(C_body_cmd)
        else:
            link.gimbal(pan_c, info["joints"][2])
        return pan_c, tilt_c, info

    vision = None
    if args.measure == "yolo":
        vision = start_yolo(args.yolo_model, cam, H)
    st = wait_for_jeep(link, clock)
    v_meas = float("nan")
    settle = 2.0

    def place(st_ref):
        """擺放飛機。預先補上「暫停等待 settle 秒」期間吉普車會開走的距離。

        2026-11-17 實驗 A：飛機 PC 的 X-Plane 在暫停中量到的吉普車速度只有 6.9 m/s，
        但紀錄中吉普車其實一直是 13.2 m/s → 暫停時多人連線位置更新不可靠。
        所以補償改用「這個 V_c 應有的速度 × 吉普車航向」，實測速度只拿來檢查筆電的 --speed 有沒有打錯。
        """
        lead = v_jeep * (settle + 0.1)
        shift = (lead * math.cos(st_ref.jpsi), lead * math.sin(st_ref.jpsi))
        place_relative_to_jeep(link, st_ref, xp, k, H, jeep_shift=shift)
        clock.sleep(settle)
        return link.read()

    def point_gimbal(st_now):
        """起始時用真值對準吉普車一次（只用在初始化，之後只靠影像）。"""
        if follow:
            g[0], g[1] = geo.los_angles(np.asarray(st_now.jeep_pos) - np.asarray(st_now.pos))
        else:
            p, t_ = geo.pointing_angles(st_now.jeep_pos, st_now.pos, st_now.psi,
                                        math.radians(st_now.pitch_deg), math.radians(st_now.roll_deg))
            g[0], g[1] = p, float(np.clip(t_, tilt_min, tilt_max))
        return command_gimbal(st_now)

    if not args.no_reset:
        clock.sleep(1.0)
        st2 = link.read()
        dtj = max(st2.t - st.t, 1e-3)
        v_meas = math.hypot(st2.jx - st.jx, st2.jy - st.jy) / dtj
        print(f"[擺放] 吉普車速度：應為 {v_jeep:.1f} m/s，暫停中粗測約 {v_meas:.1f} m/s（暫停時不準，只供參考）")
        if v_meas > 1.5 * v_jeep:
            print(f"[擺放] ⚠️ 吉普車比 {v_jeep:.1f} m/s 快很多：筆電的 jeep_controller.py --speed 可能打錯")
        st = place(st2)
        print(f"[擺放] 飛機離地 {st.agl:.0f} m（目標 {H:.0f} m）")
    pan, tilt, ginfo = point_gimbal(st)
    if vision is not None:
        t_wait = clock.now()
        clock.sleep(0.2)                     # 讓 X-Plane 先畫出雲台轉好的畫面
        wait_first_detection(vision, clock, since=clock.now())
        extra = clock.now() - t_wait
        if not args.no_reset and extra > 0.3:
            # 等 YOLO 的時間模擬是暫停的，但吉普車照樣在開 → 重新擺放，起始距離才會和 truth 實驗一樣
            # （2026-11-17 YOLO 版 A：等了約 0.5 s，起始 258 m 而不是 255 m；遠距離可能要等到 8 s ≈ 150 m）
            print(f"[擺放] 等 YOLO 用了 {extra:.1f} s，吉普車又開了約 {v_jeep * extra:.0f} m → 重新擺放")
            st = place(link.read())
            pan, tilt, ginfo = point_gimbal(st)
        frames0 = vision[1].frame_count
    poses = PoseHistory()
    rec = Recorder()
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = output_dir(xp, f"yang_{args.controller}_{args.measure}_{xp['scenario']['start']}"
                             f"_vc{xp['scenario']['jeep_speed_paper']:g}{'_enh' if enhanced else ''}")
    print(f"[Yang X-Plane] 紀錄存到 {out}")

    overlay = status = None
    # 預設每次都錄影（xp_params.yaml 的 loop.record_overlay），--no-record 可關掉
    record = (args.record or xp["loop"].get("record_overlay", True)) and not args.no_record
    if (args.show or record) and not args.dry_run:
        from .overlay import OverlayStatus, OverlayThread, pick_window_position, screen_grabber
        region = {"top": xp["camera"]["crop_top"], "left": 0, "width": cam.width, "height": cam.height}
        show = args.show
        wx, wy = args.show_x, args.show_y
        if show and (wx is None or wy is None):
            try:
                pos, desc = pick_window_position(region)
            except Exception as exc:  # noqa: BLE001
                pos, desc = None, f"讀不到螢幕資訊：{exc}"
            print(f"[overlay] 偵測到的螢幕：{desc}")
            if pos is None:
                if vision is not None:
                    print("[overlay] ⚠️ 找不到 X-Plane 以外的螢幕：YOLO 模式下視窗會被截進畫面，所以不開視窗"
                          "（有 --record 仍會錄影）。可用 --show-x --show-y 自己指定位置。")
                    show = False
                pos = (0, 0)
            wx = pos[0] if wx is None else wx
            wy = pos[1] if wy is None else wy
        status = OverlayStatus()
        status.update(R_M=params["R_M"])
        if vision is not None:
            source = vision[1].get_debug_frame
        else:
            source = screen_grabber(region)
        overlay = OverlayThread(status, source, cam.cx, cam.cy, show=show,
                                record_path=(out / "overlay.mp4") if record else None,
                                window_x=wx or 0, window_y=wy or 0)
        overlay.start()
        if show:
            print(f"[overlay] 顯示{'＋錄影' if record else ''}已啟動，視窗位置 (x, y) = ({wx}, {wy})")
        else:
            print(f"[overlay] 只錄影（不開視窗），存到 {out / 'overlay.mp4'}")

    u_cmd, u_prev, U, switcher = params["u_psi_init"], params["u_psi_init"], None, None
    s1, s2_last = (0.0, 0.0), (0.0, 0.0)
    rates = (0.0, 0.0)
    mode, n1, solve_ms, detected = "-", -1, np.nan, 0.0
    yolo_ts = float("nan")
    u2 = v2 = u3 = v3 = g1 = np.nan
    meas_valid, predicted = 0, 0
    if lost_handling:
        # 起始時雲台已對準吉普車（目標在畫面中心）：用它當 s2 的初值，第一次偵測前的推算才有正確的起點
        # （2026-11-18 改良版 YOLO E1：第一步還沒偵測到時從 s2 = (0, 0) 推算，畫面特徵在 t = 0 跳到 v1 ≈ 4972 px）
        C_H0 = geo.rows_body_to_level(C_body_cmd, math.radians(st.pitch_deg), math.radians(st.roll_deg))
        u2_0, v2_0, _, _, ok0 = geo.ideal_and_down_from_ray(geo.ray_level_from_pixel(0.0, 0.0, C_H0, f), alpha, f)
        if ok0:
            s2_last = (u2_0, v2_0)
    t0 = clock.now()
    next_mpc = t0
    last_tick = t0
    last_mpc_t = t0
    last_meas_t = t0                 # 最後一次有效量測的時間（跟丟計時）
    last_meas_ts = -1.0              # 最後用過的 YOLO 影像時間戳（同一張不重複用）
    tick = 1.0 / xp["loop"]["control_hz"]
    min_agl = xp["loop"]["min_agl_ratio"] * H
    link.pause(False)
    link.message("Yang 2024 tracking: running")
    try:
        while True:
            now = clock.now()
            if now - t0 >= T:
                break
            dt = now - last_tick
            last_tick = now
            st = link.read()
            if st.agl < min_agl:
                print(f"[Yang X-Plane] 離地 {st.agl:.0f} m < {min_agl:.0f} m，暫停並結束")
                break
            if st.airspeed < 5.0 and now - t0 < 1.0:
                # 剛解除暫停時 X-Plane 有時回傳空速 0：跳過這一圈，避免油門被推到全開
                clock.sleep(tick - (clock.now() - now))
                continue
            pitch, roll = math.radians(st.pitch_deg), math.radians(st.roll_deg)
            # 這一刻 X-Plane 實際的相機姿態 = 上一圈送出的頭部角度（相對機身）× 這一刻的飛機姿態
            C_H = geo.rows_body_to_level(C_body_cmd, pitch, roll)
            poses.add(now, C_H, st.psi)
            if st.jeep_ok:
                ut1, vt1, depth = geo.project_rows(st.jeep_pos, st.pos, st.psi, C_H, f)
            else:
                ut1, vt1, depth = np.nan, np.nan, -1.0

            mpc_tick = now >= next_mpc
            if mpc_tick:
                next_mpc = max(next_mpc + dt_mpc, now + 0.5 * dt_mpc)
                dt_m = now - last_mpc_t
                last_mpc_t = now
                # ---------- 1. 量測
                meas, C_cap, psi_cap = None, C_H, st.psi
                if args.measure == "truth":
                    ok_meas = depth > 0 and (not fov_check or cam.in_image(ut1, vt1))
                    if ok_meas:
                        meas = (ut1, vt1)
                    detected = float(ok_meas)
                    yolo_ts = float("nan")
                else:
                    detector, shared, _ = vision
                    detector.update_altitude(st.agl)
                    res, _ = shared.get_latest()
                    detected = float(bool(res.detected))
                    yolo_ts = float(res.timestamp) if res.timestamp else float("nan")
                    fresh = (not lost_handling) or res.timestamp > last_meas_ts
                    if res.detected and fresh:
                        hist = poses.at(res.timestamp)
                        if hist is not None:
                            C_cap, psi_cap = hist[1], hist[2]
                        meas = cam.to_centered(res.raw_center_x, res.raw_center_y)
                        last_meas_ts = res.timestamp
                los_w = None
                if meas is not None:
                    # ---------- 2a. 有量測：換成 Ideal State（Eq. 7、15）
                    s1 = meas
                    ray = geo.ray_level_from_pixel(s1[0], s1[1], C_cap, f)
                    u2, v2, u3, v3, ok = geo.ideal_and_down_from_ray(ray, alpha, f)
                    if ok:
                        s2_last = (u2, v2)
                    u2, v2 = s2_last
                    los_w = geo.rot_z(psi_cap) @ ray
                    last_meas_t = now
                    meas_valid, predicted = 1, 0
                elif lost_handling:
                    # ---------- 2b. 偵測不到（改良版）：用 Eq. 10 + 實測航向變化率推算 s2（已知的自身運動，不做定位）
                    u2p, v2p = s2_last
                    z = depth_of(v2p)
                    if np.isfinite(z) and z > 0:
                        du, dv = geo.image_model(u2p, v2p, st.psi_rate, V, alpha, f, z)
                        s2_last = (u2p + du * dt_m, v2p + dv * dt_m)
                    u2, v2 = s2_last
                    if v2 * math.cos(alpha) + f * math.sin(alpha) > 1e-6:
                        u3, v3 = geo.ideal_to_down(u2, v2, alpha, f)
                    ray = geo.ideal_ray_level(u2, v2, f, alpha)
                    los_w = geo.rot_z(st.psi) @ ray
                    c = C_H @ ray                         # 推算的目標在目前畫面上的位置（只供顯示、原版雲台律用）
                    s1 = (f * c[0] / c[2], f * c[1] / c[2]) if c[2] > 1e-6 else (np.nan, np.nan)
                    meas_valid, predicted = 0, 1
                else:
                    # ---------- 2c. 偵測不到（原版）：沿用上一筆 s1，配上目前的相機姿態
                    ray = geo.ray_level_from_pixel(s1[0], s1[1], C_H, f)
                    u2, v2, u3, v3, ok = geo.ideal_and_down_from_ray(ray, alpha, f)
                    if ok:
                        s2_last = (u2, v2)
                    u2, v2 = s2_last
                    meas_valid, predicted = 0, 0
                g1 = geo.perception_g(u3, v3, f, params["R_M"], H)

                # ---------- 3. 控制器（和數值模擬同一份）
                if args.controller == "lsm":
                    u_cmd = lsm_u_psi(u2, v2, V, alpha, f, params)
                    mode = "LSM"
                else:
                    if args.controller == "p1soft":
                        mode = "J1"
                    else:
                        if switcher is None:
                            switcher = Switcher(params, g1)
                        else:
                            switcher.update(g1)
                        mode, n1 = switcher.mode, switcher.n1
                    U0 = mpc.lsm_warm_start(u2, v2, u_prev) if U is None else YangMPC.shift(U)
                    U, info = mpc.solve(mode, u2, v2, u_prev, U0)
                    u_cmd, solve_ms = float(U[0]), info["time"] * 1e3
                u_prev = u_cmd

                # ---------- 4. 雲台律（Eq. 9）
                if follow:
                    # 世界座標的視線：每個 MPC 週期把視線往目標方向拉 λ 倍（= Eq. 9 的比例律，在增穩座標下做）
                    if los_w is not None:
                        az_t, el_t = geo.los_angles(los_w)
                        rates = (float(np.clip(lam_u * geo.wrap(az_t - g[0]) / dt_mpc, -rate_max, rate_max)),
                                 float(np.clip(lam_v * (el_t - g[1]) / dt_mpc, -rate_max, rate_max)))
                    else:
                        rates = (0.0, 0.0)
                elif np.isfinite(s1[0]) and np.isfinite(s1[1]):
                    # 原版：每個 MPC 週期把目標拉回中心的角度，換成角速度
                    rates = (float(np.clip(math.atan(-lam_u * s1[0] / f) / dt_mpc, -rate_max, rate_max)),
                             float(np.clip(math.atan(-lam_v * s1[1] / f) / dt_mpc, -rate_max, rate_max)))
                else:
                    rates = (0.0, 0.0)

            lost_s = now - last_meas_t
            if status is not None and mpc_tick:
                upd = dict(t=now - t0, mode=mode, n1=n1, u_cmd=u_cmd, roll=st.roll_deg, u2=u2, v2=v2,
                           R=math.hypot(st.jx - st.x, st.jy - st.y) if st.jeep_ok else np.nan,
                           u1_truth=ut1, v1_truth=vt1, bbox=None,
                           pred_uv=s1 if predicted else None,
                           gimbal_text=f"gimbal: {gimbal.mode}"
                                       f"{'  SATURATED' if ginfo.get('saturated') else ''}"
                                       f"{'  (flipped)' if ginfo.get('flipped') else ''}")
                lost_txt = f"  LOST {lost_s:.1f} s (predicting)" if predicted else ""
                if vision is not None:
                    res, _ = vision[1].get_latest()
                    upd.update(bbox=res.bbox if res.detected else None, conf=res.confidence,
                               vision_text=f"YOLO: {'detected' if res.detected else 'MISSED'}  "
                                           f"{vision[1].vision_fps:.0f} Hz  age {(now - res.timestamp) * 1e3:.0f} ms"
                                           + lost_txt)
                else:
                    upd.update(vision_text=f"measure: truth{'' if meas_valid else '  OUT OF IMAGE'}" + lost_txt)
                status.update(**upd)

            # 雲台：依真實經過時間積分，再換成相機姿態送出
            if follow:
                g[0] = geo.wrap(g[0] + rates[0] * dt)
                g[1] = float(np.clip(g[1] + rates[1] * dt, -math.pi / 2, math.pi / 4))
            else:
                g[0] = geo.wrap(g[0] + rates[0] * dt)
                g[1] = float(np.clip(g[1] + rates[1] * dt, tilt_min, tilt_max))
            pan, tilt, ginfo = command_gimbal(st)

            elev, ail, rud, thr = autopilot.update(st, u_cmd, dt)
            link.controls(elev, ail, rud, thr)

            jy_, jr_, jp_ = ginfo["joints"]
            rec.add(t=now, mpc_tick=int(mpc_tick), x=st.x, y=st.y, agl=st.agl, psi=st.psi,
                    heading=st.heading_deg, psi_rate=st.psi_rate,
                    psi_rate_coord=-9.81 * math.tan(math.radians(st.roll_deg)) / max(st.airspeed, 1.0),
                    roll=st.roll_deg, pitch=st.pitch_deg,
                    airspeed=st.airspeed, jx=st.jx, jy=st.jy, jeep_ok=int(st.jeep_ok),
                    R_truth=math.hypot(st.jx - st.x, st.jy - st.y) if st.jeep_ok else np.nan,
                    u_cmd=u_cmd, phi_cmd=autopilot.last["phi_cmd"], theta_cmd=autopilot.last["theta_cmd"],
                    elevator=elev, aileron=ail, rudder=rud, throttle=thr,
                    mode=mode, n1=n1, solve_ms=solve_ms, detected=detected, yolo_ts=yolo_ts,
                    meas_valid=meas_valid, predicted=predicted, lost_s=lost_s,
                    u1=s1[0], v1=s1[1], u1_truth=ut1, v1_truth=vt1, u2=u2, v2=v2, u3=u3, v3=v3, g1=g1,
                    pan=pan, tilt=tilt, pan_rate=rates[0], tilt_rate=rates[1],
                    joint_yaw=jy_, joint_roll=jr_, joint_pitch=jp_,
                    gimbal_sat=int(ginfo["saturated"]), gimbal_flip=int(ginfo["flipped"]),
                    cam_roll=math.degrees(ginfo["cam_roll"]))
            clock.sleep(tick - (clock.now() - now))
    except KeyboardInterrupt:
        print("\n[Yang X-Plane] 使用者中斷")
    finally:
        link.pause(True)
        if overlay is not None:
            overlay.stop()
        yolo_frames = None
        if vision is not None:
            yolo_frames = vision[1].frame_count - frames0
            vision[2].stop()
        rec.save_csv(out / "log.csv")
        if rec.rows:
            summary = summarize(rec, params, k, dt_mpc)
            summary.update(profile=xp["profile"], controller=args.controller, measure=args.measure,
                           setup="enhanced" if enhanced else "yang_original", gimbal_mode=gimbal.mode,
                           lost_handling=lost_handling, truth_fov_check=fov_check,
                           start=xp["scenario"]["start"], vc_paper=xp["scenario"]["jeep_speed_paper"],
                           jeep_speed_target=v_jeep, jeep_speed_paused_probe=v_meas,
                           **run_checks(rec, params, v_jeep, start_distance(xp, k, params)))
            if yolo_frames is not None and summary["duration_s"] > 0:
                summary["yolo_fps"] = yolo_frames / summary["duration_s"]   # YOLO 實際處理張數（yolo_hz 受 MPC 取樣上限 12.5 Hz）
            save_json(out / "summary.json", {"summary": summary, "params": params, "xp": xp})
            plot(rec, params, out / "overview.png",
                 f"Yang 2024 in X-Plane: {args.controller}, {args.measure}, start = {xp['scenario']['start']}, "
                 f"{'enhanced (gimbal ' + gimbal.mode + ', lost handling)' if enhanced else 'Yang original'}")
            for key in ("duration_s", "loop_hz", "mpc_hz", "rmse_s2_norm_x537", "R_max_m", "time_outside_RM_s",
                        "u_sat_ratio", "psi_rate_tracking_rms", "agl_max_err_m", "roll_max_deg",
                        "elevator_std", "yolo_detect_ratio", "yolo_fps", "yolo_age_ms_mean", "yolo_err_px_rms",
                        "meas_valid_ratio", "predicted_ratio", "lost_max_s", "lost_episodes_over_0p5s",
                        "gimbal_sat_ratio", "cam_roll_max_deg"):
                if summary.get(key) is None:
                    continue
                print(f"  {key:24s} {summary[key]}")
        print(f"[Yang X-Plane] 結束，結果在 {out}")
    return out


if __name__ == "__main__":
    main()
