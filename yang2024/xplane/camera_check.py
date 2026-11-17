"""確認相機模型：在 X-Plane 截圖上畫出「用吉普車真值算出的位置」，看有沒有對準吉普車。

Stage 3（真值特徵）完全依賴這個相機模型，所以第一次一定要先做。它同時檢查：
  焦距 f、主點（擷取畫面切掉上方 30 px）、雲台角度方向（pan 往左為正 ↔ pilots_head_psi 往右為正）、
  學長的 8.5° 俯仰校正、座標與航向換算、多人連線的吉普車位置。

做法：把飛機擺到論文起始位置並**暫停**，雲台依序轉到幾組角度，每組截一張圖：
  綠色十字 = 程式算出的吉普車位置；有給 --yolo-model 時，紅框 = YOLO 偵測。
  十字落在吉普車上就代表模型正確。方向檢查：pan +5°（雲台往左）時吉普車往畫面右邊移，
  約 f·tan5°·cos(俯角)（俯角 40–50° 時約 95–110 px）；tilt −5°（往下看）時吉普車往上移約 145 px。

用法（飛機 PC，repo 根目錄；筆電的吉普車停在跑道上即可，不用開）：
    python -m yang2024.xplane.camera_check
    python -m yang2024.xplane.camera_check --yolo-model <best.pt 路徑>
    python -m yang2024.xplane.camera_check --dry-run     # 不截圖，只測程式
"""
import argparse
import math
from pathlib import Path

import numpy as np

from .. import geometry as geo
from .camera import XPCamera
from .common import output_dir, save_json, setup, wait_for_jeep
from .xp_link import place_relative_to_jeep

OFFSETS_DEG = [(0.0, 0.0), (5.0, 0.0), (-5.0, 0.0), (0.0, -5.0), (0.0, 5.0)]  # (pan, tilt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-model", default=None)
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    link, clock, params, xp, k = setup(dry_run=args.dry_run)
    cam = XPCamera(xp["camera"])
    out = Path(args.out) if args.out else output_dir(xp, "camera_check")
    out.mkdir(parents=True, exist_ok=True)

    st = wait_for_jeep(link, clock)
    if not args.no_reset:
        place_relative_to_jeep(link, st, xp, k, params["H"])
        clock.sleep(1.0)
        link.pause(False)   # X-Plane 暫停時可能不會套用新位置：放開 1 秒讓它生效
        clock.sleep(1.0)
    link.pause(True)
    clock.sleep(1.0)
    st = link.read()
    dist = math.dist(st.pos, st.jeep_pos)
    print(f"[camera check] 飛機離地 {st.agl:.0f} m、和吉普車的直線距離 {dist:.0f} m（預期約 "
          f"{params['H']:.0f} m 與 {math.hypot(params['H'], 85.0 * k['k_L']):.0f} m）")
    if not args.no_reset and st.agl < 0.5 * params["H"]:
        print("[camera check] ⚠️ 飛機沒有被移到空中，擺放可能沒生效；結果請截圖給我")

    grab = detector = None
    if not args.dry_run:
        import cv2
        import mss
        sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        region = {"top": xp["camera"]["crop_top"], "left": 0,
                  "width": cam.width, "height": cam.height}

        def grab():
            return np.ascontiguousarray(np.array(sct.grab(region))[:, :, :3])
        if args.yolo_model:
            print("[camera check] 載入 YOLO 模型……")
            from ultralytics import YOLO
            detector = YOLO(args.yolo_model)

    rows = []
    for dp, dt_ in OFFSETS_DEG:
        st = link.read()
        pitch, roll = math.radians(st.pitch_deg), math.radians(st.roll_deg)
        pan, tilt = geo.pointing_angles(st.jeep_pos, st.pos, st.psi, pitch, roll)
        pan, tilt = pan + math.radians(dp), tilt + math.radians(dt_)
        link.gimbal(pan, tilt)
        clock.sleep(1.0)
        st = link.read()
        u, v, depth = cam.project_truth(st, pan, tilt)
        px, py = cam.to_pixel(u, v)
        row = {"pan_offset_deg": dp, "tilt_offset_deg": dt_, "pred_px": px, "pred_py": py,
               "slant_m": float(math.dist(st.pos, st.jeep_pos)), "visible": bool(depth > 0)}
        if grab is not None:
            img = grab()
            if detector is not None:
                r = detector.predict(img, conf=0.05, verbose=False)[0]
                if len(r.boxes):
                    b = max(r.boxes, key=lambda bb: float(bb.conf[0]))
                    x1, y1, x2, y2 = (int(round(c)) for c in b.xyxy[0].tolist())
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    row.update(yolo_px=(x1 + x2) / 2, yolo_py=(y1 + y2) / 2, yolo_conf=float(b.conf[0]),
                               err_px=float(math.hypot((x1 + x2) / 2 - px, (y1 + y2) / 2 - py)))
            c = (int(round(px)), int(round(py)))
            cv2.drawMarker(img, c, (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
            cv2.circle(img, c, 25, (0, 255, 0), 1)
            cv2.putText(img, f"pan {dp:+.0f} deg, tilt {dt_:+.0f} deg", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(str(out / f"check_pan{dp:+.0f}_tilt{dt_:+.0f}.png"), img)
        rows.append(row)
        print(f"  pan {dp:+.0f}°, tilt {dt_:+.0f}° → 預測像素 ({px:7.1f}, {py:7.1f})"
              + (f"，YOLO ({row['yolo_px']:.1f}, {row['yolo_py']:.1f})，差 {row['err_px']:.1f} px"
                 if "yolo_px" in row else ""))
    save_json(out / "summary.json", {"rows": rows, "f": cam.f, "cx": cam.cx, "cy": cam.cy})
    print(f"[camera check] 截圖與結果在 {out}")
    return out, rows


if __name__ == "__main__":
    main()
