"""
Gimbal controller + ray-projection distance estimator.

- SEARCH: gimbal tracks target to image center
- PRE_DIVE: vision-driven LoS stabilization while aircraft pitches down
- DIVE: gimbal locked
"""

import numpy as np
import math
from collections import deque


# ==================== Rotation utilities ====================

def _Rx(phi_deg):
    """Rotation about X (roll)."""
    phi = math.radians(phi_deg)
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(theta_deg):
    """Rotation about Y (pitch)."""
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def _Rz(psi_deg):
    """Rotation about Z (yaw)."""
    psi = math.radians(psi_deg)
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


class RayProjectionDistanceEstimator:
    """
    Geometric ray-projection range estimator.

    pixel (u,v) -> camera ray -> rotate to NED -> intersect ground plane
    -> depth = altitude / ray_z. Needs only intrinsics + attitude + altitude
    (no target size).

    Note on resolutions: X-Plane renders at render_width x render_height;
    the screenshot crops crop_top pixels off the top. fx/fy come from the
    render resolution; cy is shifted by crop_top.
    """

    def __init__(self,
                 fov_horizontal=60.0,
                 fov_vertical=None,
                 screen_width=1920,
                 screen_height=1010,
                 render_width=None,
                 render_height=None,
                 crop_top=30,
                 ema_alpha=0.3,
                 outlier_factor=2.0,
                 gimbal_pitch_offset_deg=0.0):
        """
        Args:
            fov_vertical: None -> derived from render aspect ratio.
            gimbal_pitch_offset_deg: constant bias between reported gimbal
                pitch and the true optical axis (cockpit default eye tilt).
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gimbal_pitch_offset_deg = gimbal_pitch_offset_deg

        if render_width is None:
            render_width = screen_width
        if render_height is None:
            render_height = screen_height + crop_top
        self.render_width = render_width
        self.render_height = render_height
        self.crop_top = crop_top

        if fov_vertical is None:
            aspect_ratio = render_width / render_height
            fov_vertical = 2.0 * math.degrees(
                math.atan(math.tan(math.radians(fov_horizontal / 2.0)) / aspect_ratio)
            )
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical

        # Intrinsics from FOV + render resolution
        self.fx = render_width / (2.0 * math.tan(math.radians(fov_horizontal / 2.0)))
        self.fy = render_height / (2.0 * math.tan(math.radians(fov_vertical / 2.0)))

        # Principal point: render center, cy shifted by crop_top
        self.cx = render_width / 2.0
        self.cy = render_height / 2.0 - crop_top

        # Gimbal -> Camera (CV) rotation
        self.R_g2c = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        self.R_c2g = self.R_g2c.T

        # Smoothing
        self.ema_alpha = ema_alpha
        self.outlier_factor = outlier_factor
        self.distance_history = deque(maxlen=15)
        self.ema_distance = None

        # Debug
        self.last_ray_world = None
        self.last_raw_slant = None
        self.last_ground_range = None

        print(f"[RayProjectionDistanceEstimator] init:")
        print(f"  render: {render_width}x{render_height}")
        print(f"  capture: {screen_width}x{screen_height} (crop_top={crop_top})")
        print(f"  FOV: {fov_horizontal:.1f} x {fov_vertical:.1f} deg")
        print(f"  intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def estimate_distance(self, pixel_u, pixel_v,
                          aircraft_roll, aircraft_pitch, aircraft_yaw_true,
                          gimbal_pitch, gimbal_yaw,
                          altitude_agl):
        """
        Estimate range to a ground target.

        Returns:
            (slant_range, ground_range, confidence); ranges None if the ray
            does not point downward.
        """
        # Pixel -> normalized camera ray
        x1 = (pixel_u - self.cx) / self.fx
        x2 = (pixel_v - self.cy) / self.fy
        Pc = np.array([x1, x2, 1.0])
        Ic = Pc / np.linalg.norm(Pc)

        # Rotation chain: Camera -> Gimbal -> Body -> NED
        gimbal_pitch_eff = gimbal_pitch - self.gimbal_pitch_offset_deg
        R_b2g = _Ry(gimbal_pitch_eff) @ _Rz(gimbal_yaw)
        R_w2b = _Rx(aircraft_roll) @ _Ry(aircraft_pitch) @ _Rz(aircraft_yaw_true)
        R_ned2cam = self.R_g2c @ R_b2g @ R_w2b
        R_cam2ned = np.linalg.inv(R_ned2cam)

        Iw = R_cam2ned @ Ic
        self.last_ray_world = Iw.copy()

        # Ground intersection: t = altitude / Iw_z (Iw_z > 0 = pointing down)
        if Iw[2] <= 0.01:
            return self._smooth(None), None, 0.0

        slant_range = altitude_agl / Iw[2]
        self.last_raw_slant = slant_range

        horizontal = math.sqrt(Iw[0]**2 + Iw[1]**2)
        ground_range = slant_range * horizontal
        self.last_ground_range = ground_range

        # Confidence: steeper ray -> more stable estimate
        confidence = np.clip(Iw[2] / 0.3, 0.0, 1.0)

        smoothed_slant = self._smooth(slant_range)

        if smoothed_slant is not None and smoothed_slant > altitude_agl:
            smoothed_ground = math.sqrt(smoothed_slant**2 - altitude_agl**2)
        else:
            smoothed_ground = ground_range

        return smoothed_slant, smoothed_ground, confidence

    def _smooth(self, raw_distance):
        """Median + outlier rejection + EMA."""
        if raw_distance is None or raw_distance <= 0:
            return self._get_current_estimate()

        if len(self.distance_history) >= 5:
            median = np.median(list(self.distance_history))
            if median > 0:
                ratio = raw_distance / median
                if ratio > self.outlier_factor or ratio < (1.0 / self.outlier_factor):
                    raw_distance = np.clip(raw_distance,
                                           median / self.outlier_factor,
                                           median * self.outlier_factor)

        self.distance_history.append(raw_distance)
        median_dist = np.median(list(self.distance_history))

        if self.ema_distance is None:
            self.ema_distance = median_dist
        else:
            self.ema_distance = (self.ema_alpha * median_dist +
                                 (1.0 - self.ema_alpha) * self.ema_distance)

        return self.ema_distance

    def _get_current_estimate(self):
        if self.ema_distance is not None:
            return self.ema_distance
        if len(self.distance_history) > 0:
            return np.median(list(self.distance_history))
        return None

    def get_ground_range(self, altitude_agl=None):
        return self.last_ground_range

    def reset(self):
        self.distance_history.clear()
        self.ema_distance = None
        self.last_ray_world = None
        self.last_raw_slant = None
        self.last_ground_range = None


class GimbalController:
    """Two-axis gimbal controller with SEARCH tracking and PRE_DIVE stabilization."""

    def __init__(self,
                 dt=0.05,
                 screen_width=1920,
                 screen_height=1010,
                 max_gimbal_pitch=60.0,
                 max_gimbal_yaw=45.0,
                 gimbal_pitch_rate=30.0,
                 gimbal_yaw_rate=45.0,
                 camera_offset_x=0.0,
                 camera_offset_y=0.0,
                 camera_offset_z=0.0,
                 xplane_pitch_calibration_deg=0.0):
        """
        Args:
            xplane_pitch_calibration_deg: constant bias between the head_the
                DREF and the true optical axis; applied only when sending to
                X-Plane so gimbal_pitch=0 means "camera along body axis"
                everywhere else.
        """
        self.dt = dt
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.center_x = screen_width / 2
        self.center_y = screen_height / 2

        self.max_pitch = max_gimbal_pitch
        self.max_yaw = max_gimbal_yaw
        self.pitch_rate = gimbal_pitch_rate
        self.yaw_rate = gimbal_yaw_rate

        self.gimbal_pitch = 0.0
        self.gimbal_yaw = 0.0

        self.camera_offset_x = camera_offset_x
        self.camera_offset_y = camera_offset_y
        self.camera_offset_z = camera_offset_z
        self.xplane_pitch_calibration_deg = float(xplane_pitch_calibration_deg)

        self.is_locked = False
        self.lock_pitch = 0.0
        self.lock_yaw = 0.0

        # Tracking PD gains
        self.track_pitch_kp = 0.05
        self.track_pitch_kd = 0.0
        self.track_yaw_kp = 0.03
        self.track_yaw_kd = 0.001

        self.prev_pitch_error = 0.0
        self.prev_yaw_error = 0.0

        # Anti-jitter: vision runs ~15 Hz vs 100 Hz control, so a repeated
        # bbox must not re-trigger the PD; error is LPF'd before derivative.
        self._last_target_x_for_pd = None
        self._last_target_y_for_pd = None
        self._error_x_lpf = 0.0
        self._error_y_lpf = 0.0
        self._prev_error_x_lpf = 0.0
        self._prev_error_y_lpf = 0.0
        self._lpf_alpha = 0.6
        self._vision_dt = 0.067   # estimated vision period (15 Hz)

        # PRE_DIVE tracking shape
        self.deadband_px = 8         # filters vision noise near center
        self.fine_threshold_px = 50  # two-stage gain switch point
        self.fine_gain_ratio = 0.4   # near-center gain ratio
        self.kff_q = 1.0             # pitch-rate feed-forward gain
        self.kff_p = 0.3             # roll-rate feed-forward gain

        # SEARCH stable-tracking parameters
        self.search_deadband_px = 3
        self.search_fine_gain_ratio = 0.3

        # SEARCH zone-based tracking:
        #   inside green: deadband; green~yellow: slow pull; outside yellow: fast pull
        self.search_green_ratio = 0.25
        self.search_yellow_ratio = 0.60
        self.search_zone_low_kp_pitch = 0.10
        self.search_zone_low_kp_yaw = 0.075
        self.search_zone_high_kp_pitch = 0.06
        self.search_zone_high_kp_yaw = 0.04
        self.search_zone_low_rate = 75.0
        self.search_zone_high_rate = 45.0

        # PRE_DIVE gimbal must be faster to keep up with the pitch-down
        self.predive_pitch_rate = 60.0
        self.predive_yaw_rate = 45.0
        self.in_predive = False

        # PRE_DIVE LoS transition state
        self._predive_los_initial = None
        self._predive_pitch_initial = None

        # Vision-driven LoS stabilization:
        #   Vision (~37 Hz): pixel error -> los_target (LPF'd)
        #   100 Hz:          los_actual <- slew -- los_target
        #   100 Hz:          gimbal_pitch = los_actual - aircraft_pitch
        self._stab_init_time = None       # also the "initialized" flag
        self._stab_init_heading = None
        self._stab_los_target = None
        self._stab_los_actual = None
        self._stab_yaw_target = None
        # Compat fields (logging only)
        self._stab_los_initial = None
        self._stab_los_final = None
        self._stab_init_gimbal_yaw = None
        self._stab_pitch_correction = 0.0
        self._stab_yaw_correction = 0.0
        self._stab_R_target = None

        # Stabilization tunables
        self._stab_deadband_px = 5
        self._stab_track_alpha = 0.3       # LPF weight per vision update
        self._stab_max_los_rate = 80.0     # deg/s slew for los_actual
        self._stab_max_yaw_rate = 60.0     # deg/s slew for yaw
        # Output-side LPF on los/yaw: smooths high-frequency command noise
        # while keeping instantaneous aircraft-pitch compensation.
        self._stab_los_smooth_alpha = 0.5  # 1.0 = off
        self._stab_yaw_smooth_alpha = 0.5
        self._stab_los_smoothed = None
        self._stab_yaw_smoothed = None
        # Anti-jitter (skip vision update on repeated bbox)
        self._stab_last_bbox_x = None
        self._stab_last_bbox_y = None
        # Outlier defense (YOLO false positives / DKF spikes)
        self._stab_sanity_x_max_px = 600.0
        self._stab_sanity_y_max_px = 350.0
        self._stab_max_target_change_per_vision_deg = 1.5
        # Vision lag compensation: pose history buffer
        self._pose_history = deque(maxlen=60)   # 100 Hz x 600 ms
        self._lag_comp_enabled = True
        # 3-frame median filter against single-frame bbox spikes
        self._err_x_history = deque(maxlen=3)
        self._err_y_history = deque(maxlen=3)
        self._median_filter_enabled = True
        # Optional end-of-PRE_DIVE blend of LoS toward dive_angle (off:
        # it fights vision tracking; MPC compensates a non-zero lock angle)
        self._stab_pitch_initial = None
        self._stab_dive_blend_enabled = False
        self._stab_dive_blend_start = 0.6

        # Field of view -> deg per pixel
        self.fov_horizontal = 60.0
        self.fov_vertical = 45.0
        self.deg_per_pixel_x = self.fov_horizontal / screen_width
        self.deg_per_pixel_y = self.fov_vertical / screen_height

        print(f"[GimbalController] init: max pitch +/-{self.max_pitch}, max yaw +/-{self.max_yaw}")

    def track_target(self, target_x, target_y, target_detected, aircraft_q=0.0, aircraft_p=0.0):
        """
        Track target to image center (SEARCH / PRE_DIVE).

        Features: deadband, two-stage gain, Q/P feed-forward, per-mode rate
        limits, repeated-bbox skip, LPF'd derivative.
        """
        if self.is_locked:
            return self.gimbal_pitch, self.gimbal_yaw, True

        # Q feed-forward (pitch rate -> gimbal pitch)
        if abs(aircraft_q) > 0.5:
            pitch_ff = -self.kff_q * aircraft_q * self.dt
            self.gimbal_pitch += pitch_ff
            self.gimbal_pitch = np.clip(self.gimbal_pitch, -self.max_pitch, 0.0)

        # P feed-forward (roll rate -> gimbal yaw), PRE_DIVE only
        if self.in_predive and abs(aircraft_p) > 1.0:
            yaw_ff = -self.kff_p * aircraft_p * self.dt
            self.gimbal_yaw += yaw_ff
            self.gimbal_yaw = np.clip(self.gimbal_yaw, -self.max_yaw, self.max_yaw)

        if not target_detected:
            return self.gimbal_pitch, self.gimbal_yaw, False

        # Skip PD on repeated bbox (feed-forward above still ran)
        is_new_bbox = (self._last_target_x_for_pd is None or
                       target_x != self._last_target_x_for_pd or
                       target_y != self._last_target_y_for_pd)
        if not is_new_bbox:
            error_x_chk = target_x - self.center_x
            error_y_chk = target_y - self.center_y
            is_centered = abs(error_x_chk) < 30 and abs(error_y_chk) < 30
            return self.gimbal_pitch, self.gimbal_yaw, is_centered
        self._last_target_x_for_pd = target_x
        self._last_target_y_for_pd = target_y

        error_x = target_x - self.center_x
        error_y = target_y - self.center_y

        _search_max_rate = self.pitch_rate  # overridden by SEARCH zone logic

        if self.in_predive:
            # PRE_DIVE: deadband + two-stage gain
            if abs(error_x) < self.deadband_px:
                error_x_ctrl = 0.0
            else:
                error_x_ctrl = error_x - np.sign(error_x) * self.deadband_px

            if abs(error_y) < self.deadband_px:
                error_y_ctrl = 0.0
            else:
                error_y_ctrl = error_y - np.sign(error_y) * self.deadband_px

            if abs(error_x) < self.fine_threshold_px:
                yaw_kp = self.track_yaw_kp * self.fine_gain_ratio
                yaw_kd = self.track_yaw_kd * self.fine_gain_ratio
            else:
                yaw_kp = self.track_yaw_kp
                yaw_kd = self.track_yaw_kd

            if abs(error_y) < self.fine_threshold_px:
                pitch_kp = self.track_pitch_kp * self.fine_gain_ratio
                pitch_kd = self.track_pitch_kd * self.fine_gain_ratio
            else:
                pitch_kp = self.track_pitch_kp
                pitch_kd = self.track_pitch_kd
        else:
            # SEARCH: zone-based tracking
            ratio_x = abs(error_x) / (self.screen_width / 2) if self.screen_width > 0 else 0
            ratio_y = abs(error_y) / (self.screen_height / 2) if self.screen_height > 0 else 0
            max_ratio = max(ratio_x, ratio_y)

            if max_ratio < self.search_green_ratio:
                # Inside green zone: normal tracking
                if abs(error_x) < self.search_deadband_px:
                    error_x_ctrl = 0.0
                else:
                    error_x_ctrl = error_x - np.sign(error_x) * self.search_deadband_px

                if abs(error_y) < self.search_deadband_px:
                    error_y_ctrl = 0.0
                else:
                    error_y_ctrl = error_y - np.sign(error_y) * self.search_deadband_px

                if abs(error_x) < self.fine_threshold_px:
                    yaw_kp = self.track_yaw_kp * self.search_fine_gain_ratio
                    yaw_kd = self.track_yaw_kd * self.search_fine_gain_ratio
                else:
                    yaw_kp = self.track_yaw_kp
                    yaw_kd = self.track_yaw_kd

                if abs(error_y) < self.fine_threshold_px:
                    pitch_kp = self.track_pitch_kp * self.search_fine_gain_ratio
                    pitch_kd = self.track_pitch_kd * self.search_fine_gain_ratio
                else:
                    pitch_kp = self.track_pitch_kp
                    pitch_kd = self.track_pitch_kd

                _search_max_rate = self.pitch_rate

            elif max_ratio < self.search_yellow_ratio:
                # Green~yellow: slow pull toward center
                if abs(error_x) < self.search_deadband_px:
                    error_x_ctrl = 0.0
                else:
                    error_x_ctrl = error_x - np.sign(error_x) * self.search_deadband_px

                if abs(error_y) < self.search_deadband_px:
                    error_y_ctrl = 0.0
                else:
                    error_y_ctrl = error_y - np.sign(error_y) * self.search_deadband_px

                pitch_kp = self.search_zone_low_kp_pitch
                pitch_kd = self.track_pitch_kd * 0.3
                yaw_kp = self.search_zone_low_kp_yaw
                yaw_kd = self.track_yaw_kd * 0.3
                _search_max_rate = self.search_zone_low_rate

            else:
                # Outside yellow: fast pull toward center
                if abs(error_x) < self.search_deadband_px:
                    error_x_ctrl = 0.0
                else:
                    error_x_ctrl = error_x - np.sign(error_x) * self.search_deadband_px

                if abs(error_y) < self.search_deadband_px:
                    error_y_ctrl = 0.0
                else:
                    error_y_ctrl = error_y - np.sign(error_y) * self.search_deadband_px

                pitch_kp = self.search_zone_high_kp_pitch
                pitch_kd = self.track_pitch_kd
                yaw_kp = self.search_zone_high_kp_yaw
                yaw_kd = self.track_yaw_kd
                _search_max_rate = self.search_zone_high_rate

        # LPF the error, then differentiate at the vision rate (px/s)
        self._error_x_lpf = (self._lpf_alpha * self._error_x_lpf +
                             (1.0 - self._lpf_alpha) * error_x_ctrl)
        self._error_y_lpf = (self._lpf_alpha * self._error_y_lpf +
                             (1.0 - self._lpf_alpha) * error_y_ctrl)
        d_error_x = (self._error_x_lpf - self._prev_error_x_lpf) / self._vision_dt
        d_error_y = (self._error_y_lpf - self._prev_error_y_lpf) / self._vision_dt
        self._prev_error_x_lpf = self._error_x_lpf
        self._prev_error_y_lpf = self._error_y_lpf

        # PD control; kd_scale maps px/s derivative back to frame-diff scale
        kd_scale = 0.1
        yaw_adjustment = yaw_kp * error_x_ctrl + (yaw_kd * kd_scale) * d_error_x
        pitch_adjustment = pitch_kp * error_y_ctrl + (pitch_kd * kd_scale) * d_error_y

        # Rate limit per mode
        if self.in_predive:
            max_yaw_change = self.predive_yaw_rate * self.dt
            max_pitch_change = self.predive_pitch_rate * self.dt
        else:
            max_yaw_change = _search_max_rate * self.dt
            max_pitch_change = _search_max_rate * self.dt

        yaw_adjustment = np.clip(yaw_adjustment, -max_yaw_change, max_yaw_change)
        pitch_adjustment = np.clip(pitch_adjustment, -max_pitch_change, max_pitch_change)

        self.gimbal_yaw += yaw_adjustment
        self.gimbal_pitch -= pitch_adjustment

        self.gimbal_yaw = np.clip(self.gimbal_yaw, -self.max_yaw, self.max_yaw)
        self.gimbal_pitch = np.clip(self.gimbal_pitch, -self.max_pitch, 0.0)

        self.prev_yaw_error = error_x_ctrl
        self.prev_pitch_error = error_y_ctrl

        centering_threshold = 30
        is_centered = abs(error_x) < centering_threshold and abs(error_y) < centering_threshold

        return self.gimbal_pitch, self.gimbal_yaw, is_centered

    def stabilize_with_tracking(self, aircraft_pitch, aircraft_roll, aircraft_heading,
                                target_x, target_y, target_detected, is_new_vision,
                                dive_angle, current_time, vision_age_ms=0.0):
        """
        PRE_DIVE vision-driven LoS stabilization.

        Vision (~37 Hz): pixel error -> target LoS estimate, LPF'd into
        los_target / yaw_target.
        100 Hz: los_actual slews to los_target;
                gimbal_pitch = los_actual - aircraft_pitch;
                gimbal_yaw = yaw_target - heading_change - roll coupling.

        Returns:
            (gimbal_pitch, gimbal_yaw, is_centered)
        """
        if self.is_locked:
            return self.gimbal_pitch, self.gimbal_yaw, True

        # Push pose to history every tick: this is the gimbal pose at the
        # time the last image was captured (needed for lag compensation).
        self._pose_history.append((current_time, aircraft_pitch,
                                    self.gimbal_pitch, self.gimbal_yaw,
                                    aircraft_heading))

        # First call: initialize targets
        if self._stab_init_time is None:
            self._stab_init_time = current_time
            self._stab_init_heading = aircraft_heading

            # Start LoS continues from the SEARCH gimbal angle
            current_los = aircraft_pitch + self.gimbal_pitch
            self._stab_los_actual = current_los

            # Seed los_target from the SEARCH residual pixel error to avoid
            # a jump at the SEARCH -> PRE_DIVE transition
            if target_detected:
                err_y0 = target_y - self.center_y
                err_x0 = target_x - self.center_x
                self._stab_los_target = current_los - err_y0 * self.deg_per_pixel_y
                self._stab_yaw_target = self.gimbal_yaw + err_x0 * self.deg_per_pixel_x
            else:
                self._stab_los_target = current_los
                self._stab_yaw_target = self.gimbal_yaw

            # Compat fields
            self._stab_los_initial = current_los
            self._stab_los_final = dive_angle
            self._stab_init_gimbal_yaw = self.gimbal_yaw
            self._stab_pitch_initial = aircraft_pitch
            self._stab_los_smoothed = self._stab_los_actual
            self._stab_yaw_smoothed = self.gimbal_yaw

            print(f"[Gimbal] PRE_DIVE stabilization init:")
            print(f"  aircraft: pitch={aircraft_pitch:.1f}, roll={aircraft_roll:.1f}, heading={aircraft_heading:.1f}")
            print(f"  gimbal: pitch={self.gimbal_pitch:.1f}, yaw={self.gimbal_yaw:.1f}")
            print(f"  start LoS: {current_los:.2f}")
            print(f"  los_target: {self._stab_los_target:.2f}")
            print(f"  yaw_target: {self._stab_yaw_target:.2f}")
            return self.gimbal_pitch, self.gimbal_yaw, True

        # Vision update: only on new frame + detection + changed bbox
        is_centered = False
        is_new_bbox = (
            getattr(self, '_stab_last_bbox_x', None) is None or
            target_x != self._stab_last_bbox_x or
            target_y != self._stab_last_bbox_y
        )
        if is_new_vision and target_detected and is_new_bbox:
            self._stab_last_bbox_x = target_x
            self._stab_last_bbox_y = target_y
        elif is_new_vision and target_detected and not is_new_bbox:
            # Repeated bbox -> skip vision update, Layer 2 still runs
            err_x_chk = target_x - self.center_x
            err_y_chk = target_y - self.center_y
            is_centered = (abs(err_x_chk) < 30 and abs(err_y_chk) < 30)

        if is_new_vision and target_detected and is_new_bbox:
            err_x = target_x - self.center_x
            err_y = target_y - self.center_y

            # Sanity gate: a physical target cannot jump half a screen in
            # one frame; reject the whole frame as an outlier
            if (abs(err_x) > self._stab_sanity_x_max_px or
                abs(err_y) > self._stab_sanity_y_max_px):
                pass
            else:
                # 3-frame median filter against single-frame bbox spikes
                if self._median_filter_enabled:
                    self._err_x_history.append(err_x)
                    self._err_y_history.append(err_y)
                    if len(self._err_y_history) >= 3:
                        err_x_f = float(np.median(list(self._err_x_history)))
                        err_y_f = float(np.median(list(self._err_y_history)))
                    else:
                        err_x_f = err_x
                        err_y_f = err_y
                else:
                    err_x_f = err_x
                    err_y_f = err_y

                # Deadband on the median-filtered error
                db = self._stab_deadband_px
                ex = 0.0 if abs(err_x_f) < db else err_x_f - np.sign(err_x_f) * db
                ey = 0.0 if abs(err_y_f) < db else err_y_f - np.sign(err_y_f) * db

                # Vision lag compensation: use the gimbal pose at capture time
                if self._lag_comp_enabled and vision_age_ms > 0 and self._pose_history:
                    t_capture = current_time - vision_age_ms / 1000.0
                    closest = min(self._pose_history,
                                  key=lambda p: abs(p[0] - t_capture))
                    _, ac_pit_cap, gp_cap, gy_cap, ac_hdg_cap = closest
                    los_at_capture = ac_pit_cap + gp_cap
                    yaw_at_capture = gy_cap
                else:
                    los_at_capture = aircraft_pitch + self.gimbal_pitch
                    yaw_at_capture = self.gimbal_yaw

                los_estimate = los_at_capture - ey * self.deg_per_pixel_y
                yaw_estimate = yaw_at_capture + ex * self.deg_per_pixel_x

                # First-order LPF toward the estimate
                a = self._stab_track_alpha
                cand_los_target = (1.0 - a) * self._stab_los_target + a * los_estimate
                cand_yaw_target = (1.0 - a) * self._stab_yaw_target + a * yaw_estimate

                # Cap per-update target change (second outlier defense)
                max_dt = self._stab_max_target_change_per_vision_deg
                d_los = np.clip(cand_los_target - self._stab_los_target, -max_dt, max_dt)
                d_yaw = np.clip(cand_yaw_target - self._stab_yaw_target, -max_dt, max_dt)
                self._stab_los_target += d_los
                self._stab_yaw_target += d_yaw

                centering_threshold = 30
                is_centered = (abs(err_x) < centering_threshold and
                              abs(err_y) < centering_threshold)

        # Optional end-of-PRE_DIVE blend toward dive_angle (disabled by default)
        if (self._stab_dive_blend_enabled and
            self._stab_pitch_initial is not None):
            pitch_range = dive_angle - self._stab_pitch_initial
            if abs(pitch_range) > 0.5:
                pitch_progress = (aircraft_pitch - self._stab_pitch_initial) / pitch_range
                pitch_progress = float(np.clip(pitch_progress, 0.0, 1.0))
                bs = self._stab_dive_blend_start
                if pitch_progress > bs:
                    x = (pitch_progress - bs) / (1.0 - bs)
                    blend = x * x * (3.0 - 2.0 * x)   # smoothstep
                    los_for_slew = ((1.0 - blend) * self._stab_los_target +
                                    blend * dive_angle)
                else:
                    los_for_slew = self._stab_los_target
            else:
                los_for_slew = self._stab_los_target
        else:
            los_for_slew = self._stab_los_target

        # 100 Hz: slew los_actual toward los_for_slew
        los_step_max = self._stab_max_los_rate * self.dt
        los_diff = los_for_slew - self._stab_los_actual
        self._stab_los_actual += np.clip(los_diff, -los_step_max, los_step_max)

        # Output-side smoothing before Layer 2 (aircraft pitch stays instantaneous)
        a_los = self._stab_los_smooth_alpha
        if self._stab_los_smoothed is None:
            self._stab_los_smoothed = self._stab_los_actual
        self._stab_los_smoothed = (
            a_los * self._stab_los_actual + (1.0 - a_los) * self._stab_los_smoothed
        )

        # Layer 2: three-axis compensation (every 10 ms)
        # Pitch: full instantaneous compensation
        self.gimbal_pitch = self._stab_los_smoothed - aircraft_pitch

        # Yaw: heading hold + slew + roll coupling
        heading_change = aircraft_heading - self._stab_init_heading
        if heading_change > 180:
            heading_change -= 360
        elif heading_change < -180:
            heading_change += 360

        yaw_target_world = self._stab_yaw_target - heading_change
        yaw_step_max = self._stab_max_yaw_rate * self.dt
        yaw_diff = yaw_target_world - self.gimbal_yaw
        self.gimbal_yaw += np.clip(yaw_diff, -yaw_step_max, yaw_step_max)

        a_yaw = self._stab_yaw_smooth_alpha
        if self._stab_yaw_smoothed is None:
            self._stab_yaw_smoothed = self.gimbal_yaw
        self._stab_yaw_smoothed = (
            a_yaw * self.gimbal_yaw + (1.0 - a_yaw) * self._stab_yaw_smoothed
        )
        self.gimbal_yaw = self._stab_yaw_smoothed

        # Roll -> yaw coupling: at gimbal pitch gp, roll phi shifts yaw by
        # ~sin(gp)*phi; 0.5 = partial compensation to avoid overcorrection
        gp_rad = math.radians(self.gimbal_pitch)
        roll_rad = math.radians(aircraft_roll)
        roll_yaw_coupling = math.degrees(math.sin(gp_rad) * roll_rad)
        self.gimbal_yaw -= roll_yaw_coupling * 0.5

        self.gimbal_pitch = np.clip(self.gimbal_pitch, -self.max_pitch, 0.0)
        self.gimbal_yaw = np.clip(self.gimbal_yaw, -self.max_yaw, self.max_yaw)

        return self.gimbal_pitch, self.gimbal_yaw, is_centered

    def lock_gimbal(self, at_zero=False, lock_pitch_deg=None, lock_yaw_deg=None):
        """
        Lock the gimbal.

        Args:
            at_zero: force lock at (0, 0).
            lock_pitch_deg: explicit lock pitch (bypasses the (-max, 0) clip).
            lock_yaw_deg: explicit lock yaw.
        """
        self.is_locked = True
        if at_zero:
            self.gimbal_pitch = 0.0
            self.gimbal_yaw = 0.0
        else:
            if lock_pitch_deg is not None:
                self.gimbal_pitch = float(lock_pitch_deg)
            if lock_yaw_deg is not None:
                self.gimbal_yaw = float(np.clip(lock_yaw_deg, -self.max_yaw, self.max_yaw))
        self.lock_pitch = self.gimbal_pitch
        self.lock_yaw = self.gimbal_yaw
        print(f"[Gimbal] locked: pitch={self.gimbal_pitch:+.1f}, yaw={self.gimbal_yaw:+.1f}")

    def unlock_gimbal(self):
        self.is_locked = False
        print(f"[Gimbal] unlocked")

    def get_xplane_head_drefs(self):
        """
        Convert gimbal angles to X-Plane head DREF values.
        The pitch calibration bias is applied here and only here, so
        gimbal_pitch=0 means "camera along body axis" everywhere else.
        """
        return {
            'head_psi': self.gimbal_yaw,
            'head_the': self.gimbal_pitch + self.xplane_pitch_calibration_deg,
            'head_x': self.camera_offset_x,
            'head_y': self.camera_offset_y,
            'head_z': self.camera_offset_z,
        }

    def set_camera_position(self, offset_x=None, offset_y=None, offset_z=None):
        if offset_x is not None:
            self.camera_offset_x = offset_x
        if offset_y is not None:
            self.camera_offset_y = offset_y
        if offset_z is not None:
            self.camera_offset_z = offset_z
        print(f"[Gimbal] camera position: X={self.camera_offset_x}m Y={self.camera_offset_y}m Z={self.camera_offset_z}m")

    def reset(self, reset_position=False):
        self.gimbal_pitch = 0.0
        self.gimbal_yaw = 0.0
        self.is_locked = False
        self.lock_pitch = 0.0
        self.lock_yaw = 0.0
        self.prev_pitch_error = 0.0
        self.prev_yaw_error = 0.0
        self.in_predive = False
        self._predive_los_initial = None
        self._predive_pitch_initial = None
        self._stab_init_time = None
        self._stab_los_initial = None
        self._stab_los_final = None
        self._stab_init_heading = None
        self._stab_init_gimbal_yaw = None
        self._stab_pitch_correction = 0.0
        self._stab_yaw_correction = 0.0
        self._stab_R_target = None

        if reset_position:
            self.camera_offset_x = 0.0
            self.camera_offset_y = 0.0
            self.camera_offset_z = 0.0

        print(f"[Gimbal] reset")

    def set_predive_mode(self, enabled):
        """Enable/disable PRE_DIVE mode; clears stabilization state on exit."""
        if not enabled and self.in_predive:
            self._predive_los_initial = None
            self._predive_pitch_initial = None
            self._stab_init_time = None
            self._stab_init_heading = None
            self._stab_los_target = None
            self._stab_los_actual = None
            self._stab_yaw_target = None
            self._pose_history.clear()
            self._err_x_history.clear()
            self._err_y_history.clear()
            self._stab_pitch_initial = None
            self._stab_los_smoothed = None
            self._stab_yaw_smoothed = None
            self._stab_last_bbox_x = None
            self._stab_last_bbox_y = None
            # Compat fields
            self._stab_los_initial = None
            self._stab_los_final = None
            self._stab_init_gimbal_yaw = None
            self._stab_pitch_correction = 0.0
            self._stab_yaw_correction = 0.0
            self._stab_R_target = None
        self.in_predive = enabled
