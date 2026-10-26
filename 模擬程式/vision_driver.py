"""
Vision pipeline: YOLO car detector + async vision thread.

Vision Thread (~15-20 Hz): capture -> YOLO -> SharedVisionState
Control Loop  (100 Hz):    read SharedVisionState -> DKF -> control
"""

import threading
import time
import numpy as np
import mss
from dataclasses import dataclass
from typing import Optional, Tuple

from ultralytics import YOLO

# SAHI (sliced detection for small targets)
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
#  Car detector
# ════════════════════════════════════════════════════════════════════

class CarDetector:
    """YOLO vehicle detector with optional SAHI slicing or ByteTrack tracking."""

    def __init__(self, model_path, screen_width=1920, screen_height=1010,
                 use_sahi=False, slice_height=640, slice_width=640,
                 overlap_ratio=0.2, use_tracker=True,
                 tracker_type="bytetrack.yaml"):
        print(f"[CarDetector] loading model: {model_path}")
        self.yolo_model = YOLO(model_path)
        self.class_names = ["jeep"]
        self.confidence_threshold = 0.6
        self.current_altitude = 100.0

        # SAHI setup
        self.use_sahi = use_sahi and SAHI_AVAILABLE
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio

        if self.use_sahi:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=0.1,  # adjusted per altitude later
                device="cuda:0",
            )
            print(f"[CarDetector] SAHI on (slice {slice_width}x{slice_height}, overlap {overlap_ratio})")
        else:
            self.sahi_model = None

        # Tracker (incompatible with SAHI)
        self.use_tracker = use_tracker and (not self.use_sahi)
        self.tracker_type = tracker_type
        self.locked_track_id = None
        if self.use_tracker:
            print(f"[CarDetector] tracker on ({tracker_type})")

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.center_x = screen_width // 2
        self.center_y = screen_height // 2

    def get_adaptive_threshold(self, altitude_agl):
        """Confidence threshold by altitude (higher altitude -> lower threshold)."""
        if altitude_agl > 150:
            return 0.05
        elif altitude_agl > 100:
            return 0.08
        elif altitude_agl > 70:
            return 0.12
        elif altitude_agl > 50:
            return 0.18
        elif altitude_agl > 35:
            return 0.25
        elif altitude_agl > 20:
            return 0.35
        else:
            return 0.45

    def update_altitude(self, altitude_agl):
        self.current_altitude = altitude_agl

    def detect_vehicle(self, frame):
        """Detect vehicle. Returns (detected, cx, cy, bbox, confidence)."""
        adaptive_threshold = self.get_adaptive_threshold(self.current_altitude)
        if self.use_sahi:
            return self._detect_with_sahi(frame, adaptive_threshold)
        return self._detect_direct(frame, adaptive_threshold)

    def _detect_direct(self, frame, conf_threshold):
        if self.use_tracker:
            return self._detect_with_tracker(frame, conf_threshold)

        results = self.yolo_model.predict(
            frame,
            conf=conf_threshold,
            verbose=False,
            device="0"
        )
        result = results[0]

        if len(result.boxes) == 0:
            return False, None, None, None, 0.0

        sorted_boxes = sorted(result.boxes, key=lambda box: box.conf[0].item(), reverse=True)
        top_box = sorted_boxes[0]

        bbox = [round(x) for x in top_box.xyxy[0].tolist()]
        confidence = top_box.conf[0].item()

        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        return True, center_x, center_y, bbox, confidence

    def _detect_with_tracker(self, frame, conf_threshold):
        """YOLO + ByteTrack: keeps a locked track ID across frames."""
        results = self.yolo_model.track(
            frame,
            conf=conf_threshold,
            persist=True,
            tracker=self.tracker_type,
            verbose=False,
            device="0",
        )
        result = results[0]

        if len(result.boxes) == 0:
            return False, None, None, None, 0.0

        chosen_box = None

        # Prefer the locked ID
        if self.locked_track_id is not None:
            for box in result.boxes:
                if box.id is not None and int(box.id[0].item()) == self.locked_track_id:
                    chosen_box = box
                    break

        # No lock or lock missing this frame -> take highest confidence and lock it
        if chosen_box is None:
            sorted_boxes = sorted(result.boxes, key=lambda b: b.conf[0].item(), reverse=True)
            chosen_box = sorted_boxes[0]
            if chosen_box.id is not None:
                new_id = int(chosen_box.id[0].item())
                if self.locked_track_id != new_id:
                    print(f"[Tracker] locked target ID: {new_id}")
                    self.locked_track_id = new_id

        bbox = [round(x) for x in chosen_box.xyxy[0].tolist()]
        confidence = chosen_box.conf[0].item()

        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        return True, center_x, center_y, bbox, confidence

    def _detect_with_sahi(self, frame, conf_threshold):
        """SAHI sliced detection: split, detect per slice, merge."""
        self.sahi_model.confidence_threshold = conf_threshold

        result = get_sliced_prediction(
            frame,
            self.sahi_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            verbose=0,
        )
        predictions = result.object_prediction_list

        if len(predictions) == 0:
            return False, None, None, None, 0.0

        best = max(predictions, key=lambda p: p.score.value)

        bbox = [
            int(best.bbox.minx),
            int(best.bbox.miny),
            int(best.bbox.maxx),
            int(best.bbox.maxy),
        ]
        confidence = best.score.value

        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        return True, center_x, center_y, bbox, confidence

    def reset(self):
        self.locked_track_id = None


# ════════════════════════════════════════════════════════════════════
#  Async vision thread
# ════════════════════════════════════════════════════════════════════

@dataclass
class VisionResult:
    """Single detection result (immutable, thread-safe to pass around)."""
    timestamp: float = 0.0
    detected: bool = False
    raw_center_x: float = 960.0
    raw_center_y: float = 505.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    bbox_width: int = 0
    bbox_height: int = 0
    confidence: float = 0.0
    frame_id: int = 0


class SharedVisionState:
    """
    Thread-safe shared vision state.
    Vision thread writes, control loop reads; lock held only for assignment.
    """

    def __init__(self, screen_width=1920, screen_height=1010):
        self._lock = threading.Lock()
        self._result = VisionResult(
            raw_center_x=screen_width / 2,
            raw_center_y=screen_height / 2,
        )
        self._new_data_available = False
        self._frame_count = 0
        self._debug_frame = None

        self._vision_fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

    def update(self, result: VisionResult, debug_frame=None):
        """Called by vision thread: write new detection result."""
        with self._lock:
            self._result = result
            self._new_data_available = True
            self._frame_count += 1
            self._debug_frame = debug_frame

            self._fps_frame_count += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._vision_fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._last_fps_time = now

    def get_latest(self) -> Tuple[VisionResult, bool]:
        """Called by control loop. Returns (result, is_new_since_last_read)."""
        with self._lock:
            result = self._result
            is_new = self._new_data_available
            self._new_data_available = False
        return result, is_new

    def get_debug_frame(self):
        with self._lock:
            return self._debug_frame

    @property
    def vision_fps(self):
        with self._lock:
            return self._vision_fps

    @property
    def frame_count(self):
        with self._lock:
            return self._frame_count


class VisionThread:
    """
    Detection thread running YOLO as fast as possible (GPU-bound, ~15-25 Hz).

    Usage:
        shared_state = SharedVisionState()
        vision = VisionThread(detector, shared_state)
        vision.start()
        ...
        vision.stop()
    """

    def __init__(self, detector, shared_state: SharedVisionState,
                 screen_width=1920, screen_height=1010):
        self.detector = detector
        self.shared_state = shared_state
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.center_x = screen_width / 2
        self.center_y = screen_height / 2

        self._thread = None
        self._running = False
        self._frame_id = 0

        # mss is not thread-safe; each thread needs its own instance
        self._sct = None
        self._monitor = {"top": 30, "left": 0, "width": screen_width, "height": screen_height}

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="VisionThread")
        self._thread.start()
        print("[VisionThread] started")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[VisionThread] stopped")

    def _run(self):
        self._sct = mss.mss()

        while self._running:
            try:
                # Capture
                frame = np.array(self._sct.grab(self._monitor))
                frame = frame[:, :, :3]  # BGRA -> BGR
                capture_time = time.time()

                # Detect
                detected, raw_cx, raw_cy, bbox, confidence = self.detector.detect_vehicle(frame)

                self._frame_id += 1

                bbox_w, bbox_h = 0, 0
                bbox_tuple = None
                if detected and bbox is not None:
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    bbox_tuple = tuple(bbox)

                result = VisionResult(
                    timestamp=capture_time,
                    detected=detected,
                    raw_center_x=raw_cx if detected else self.center_x,
                    raw_center_y=raw_cy if detected else self.center_y,
                    bbox=bbox_tuple,
                    bbox_width=bbox_w,
                    bbox_height=bbox_h,
                    confidence=confidence if detected else 0.0,
                    frame_id=self._frame_id,
                )

                self.shared_state.update(result, debug_frame=frame)

            except Exception as e:
                print(f"[VisionThread] error: {e}")
                time.sleep(0.01)

    @property
    def is_running(self):
        return self._running
