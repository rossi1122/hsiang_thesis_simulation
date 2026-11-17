"""即時顯示視窗（類似學長 main.py 的 'Async Dive Attack' 視窗），可同時錄成影片。

畫面內容
  - 綠色十字圈：用真值投影算出的目標位置（truth）
  - 紅框：YOLO 偵測框（只有 --measure yolo 時有）
  - 左上文字：MPC 模式（J1 / J2 / Jg，顏色不同）、計數器 n1、水平距離 R、u_psi、坡度、YOLO 狀態

在獨立執行緒中執行，不拖慢 50 Hz 控制迴圈（約 8 Hz 更新）。
⚠️ YOLO 是截取主螢幕的 X-Plane 畫面：這個視窗一定要放在「另一個螢幕」，否則會被截進去給 YOLO 看到。
"""
import threading
import time

import numpy as np

MODE_COLOR = {"J1": (0, 220, 0), "J2": (0, 220, 255), "Jg": (0, 0, 255), "LSM": (255, 200, 0)}  # BGR


class OverlayStatus:
    """控制迴圈寫入、顯示執行緒讀取的最新狀態。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._d = {}

    def update(self, **kw):
        with self._lock:
            self._d.update(kw)

    def get(self):
        with self._lock:
            return dict(self._d)


def draw(frame, s, cx, cy, out_size=(1280, 672)):
    """在截圖上畫出狀態，回傳縮小後的影像（BGR）。frame：H×W×3 uint8。"""
    import cv2
    img = np.ascontiguousarray(frame[:, :, :3]).copy()
    h, w = img.shape[:2]
    # 畫面中心
    cv2.drawMarker(img, (int(cx), int(cy)), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
    # 真值投影
    ut, vt = s.get("u1_truth", np.nan), s.get("v1_truth", np.nan)
    if np.isfinite(ut) and np.isfinite(vt):
        p = (int(round(cx + ut)), int(round(cy + vt)))
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img, p, 25, (0, 255, 0), 2)
            cv2.drawMarker(img, p, (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
    # YOLO 偵測框
    bbox = s.get("bbox")
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(c)) for c in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"{s.get('conf', 0):.2f}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # 文字
    mode = s.get("mode", "-")
    color = MODE_COLOR.get(mode, (255, 255, 255))
    lines = [
        (f"MODE: {mode}   n1 = {s.get('n1', -1)}", color, 1.4),
        (f"t = {s.get('t', 0):5.1f} s   R = {s.get('R', np.nan):5.0f} m  (R_M {s.get('R_M', 0):.0f} m)",
         (0, 0, 255) if s.get("R", 0) > s.get("R_M", 1e9) else (255, 255, 255), 0.9),
        (f"u_psi = {s.get('u_cmd', 0):+.3f} rad/s   roll = {s.get('roll', 0):+5.1f} deg", (255, 255, 255), 0.9),
        (f"s2 = ({s.get('u2', np.nan):+6.0f}, {s.get('v2', np.nan):+6.0f}) px", (255, 255, 255), 0.9),
        (s.get("vision_text", "measure: truth"), (0, 200, 255), 0.9),
    ]
    y = 45
    for text, col, scale in lines:
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)
        y += int(42 * scale) + 10
    cv2.putText(img, "green = truth projection   red = YOLO", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cv2.resize(img, out_size)


class OverlayThread:
    """frame_source()：回傳最新截圖（或 None）。show：開視窗；record_path：存 mp4。"""

    def __init__(self, status, frame_source, cx, cy, show=True, record_path=None, window_x=1920, window_y=0,
                 hz=8.0):
        self.status, self.frame_source = status, frame_source
        self.cx, self.cy = cx, cy
        self.show, self.record_path = show, record_path
        self.window_x, self.window_y, self.period = window_x, window_y, 1.0 / hz
        self._running = False
        self._thread = None
        self.frames = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="Overlay")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self):
        import cv2
        name = "Yang 2024 X-Plane (overlay)"
        writer = None
        if self.show:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 1280, 672)
            cv2.moveWindow(name, self.window_x, self.window_y)
            try:
                cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)   # 保持在最上層
            except Exception:  # noqa: BLE001 — 舊版 OpenCV 沒有這個屬性
                pass
        try:
            while self._running:
                t_start = time.time()
                frame = self.frame_source()
                if frame is not None:
                    img = draw(frame, self.status.get(), self.cx, self.cy)
                    if self.record_path is not None:
                        if writer is None:
                            writer = cv2.VideoWriter(str(self.record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     1.0 / self.period, (img.shape[1], img.shape[0]))
                        writer.write(img)
                    if self.show:
                        cv2.imshow(name, img)
                        cv2.waitKey(1)
                    self.frames += 1
                time.sleep(max(0.0, self.period - (time.time() - t_start)))
        except Exception as exc:  # noqa: BLE001 — 顯示失敗不能影響控制
            print(f"\n[overlay] 顯示停止：{exc}")
        finally:
            if writer is not None:
                writer.release()
            if self.show:
                try:
                    cv2.destroyWindow(name)
                except Exception:  # noqa: BLE001
                    pass


def pick_window_position(capture_region, monitors=None):
    """選一個「不是 X-Plane 那個螢幕」的螢幕放顯示視窗。回傳 (x, y, 說明文字)；只有一個螢幕時回傳 None。

    monitors：mss 的螢幕清單（第 0 個是全部螢幕合起來，之後每個是一個實體螢幕，座標是實際像素）。
    """
    if monitors is None:
        import mss
        with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
            monitors = list(sct.monitors)
    screens = monitors[1:]
    cx = capture_region["left"] + capture_region["width"] / 2
    cy = capture_region["top"] + capture_region["height"] / 2

    def contains(m, x, y):
        return m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]
    desc = "、".join(f"#{i + 1}(x={m['left']}, y={m['top']}, {m['width']}×{m['height']})"
                     for i, m in enumerate(screens))
    others = [m for m in screens if not contains(m, cx, cy)]
    if not others:
        return None, desc
    m = others[0]
    return (m["left"] + 20, m["top"] + 40), desc


def screen_grabber(region):
    """truth 模式沒有 YOLO 截圖：自己截（mss 每個執行緒要有自己的實例）。"""
    local = threading.local()

    def grab():
        import mss
        if not hasattr(local, "sct"):
            local.sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        return np.array(local.sct.grab(region))[:, :, :3]
    return grab
