"""
PID dive controller: image-error PID outer loop + V11.4 cascade inner loop.

Replaces the 3DOF MPC outer loop of mpc_controller.MPCDiveController. The
attitude setpoint now comes directly from the measured bbox center (the TVS
signal), active for the whole DIVE instead of only the terminal phase.

Control law (100 Hz, every tick):

  e_lat = u_norm                  # bbox right of center -> positive
  e_lon = -v_norm                 # bbox below center -> nose down -> negative

  pitch_corr   = Kp_lon e_lon + Ki_lon INT(e_lon) + Kd_lon d(e_lon_f)/dt
  heading_corr = Kp_lat e_lat + Ki_lat INT(e_lat) + Kd_lat d(e_lat_f)/dt

  pitch_cmd   = clip(pitch_base + pitch_corr, pitch_min, pitch_max)
  heading_cmd = heading_ref + clip(heading_corr, +/- heading_max_dev)

  -> V11.4 cascade inner loop -> (rudder, elevator, aileron)

pitch_base blends from the pitch at DIVE entry to dive_pitch_target_deg over
entry_blend_time so the PRE_DIVE overshoot does not become a step pull-up
command at the transition.

The target KF, drift estimator and ray estimator still run (gimbal aiming,
trigger distance, logging) but no longer feed the control law.

Assumes the gimbal is locked at (0, 0) at DIVE entry, which main.py does via
lock_gimbal(at_zero=True). The u/v -> heading/pitch sign convention and gains
depend on it; see heading_gain_gimbal_comp for a non-zero lock angle.

Sign convention (gimbal at 0,0):
  u_norm > 0 -> bbox right of center -> heading right (positive)
  v_norm > 0 -> bbox below center    -> pitch more negative
"""

import math
import time
import numpy as np
from inner_loop_controller import (
    init_dive_inner_loop, compute_inner_loop,
    apply_rudder_slew, apply_aileron_lpf,
)
from collections import deque
from dataclasses import dataclass

# Base state machine + support classes are reused from mpc_controller.
# (This pulls in casadi at import time. If you want to drop that dependency,
# move GimbalIntegratedDiveController / TargetKalmanFilter / DriftEstimator /
# _DepthEMA / compute_R_b2c into a shared module and import from there.)
from mpc_controller import (
    GimbalIntegratedDiveController,
    DriftEstimator,
    TargetKalmanFilter,
    _DepthEMA,
    compute_R_b2c,
)


# ══════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════

@dataclass
class DivePIDParams:
    """Image-error PID outer loop parameters."""

    # ── Mission / bounds (same names as DiveMPCParams; main.py sets these) ──
    dive_pitch_target_deg: float = -30.0
    pitch_cmd_min_deg:     float = -60.0
    pitch_cmd_max_deg:     float = 20.0
    heading_cmd_max_dev_deg: float = 12.0

    # Pitch ~ gamma + alpha_nom at the dive operating point (0 = no comp)
    alpha_nominal_deg: float = 0.0

    # ── Longitudinal PID (v_norm -> pitch correction, deg) ──
    # Kp inherited from the old tvs_K_lon_deg. Ki is the new part: it removes
    # the standing image bias that the old err-driven beta floor was patching.
    Kp_lon: float = 50.0
    Ki_lon: float = 1.5      # deg per (unit error * s)
    Kd_lon: float = 0.0      # start at 0; bbox center is noisy
    I_lon_max_deg: float = 8.0     # integrator output clamp

    # ── Lateral PID (u_norm -> heading offset, deg) ──
    Kp_lat: float = 6.0
    Ki_lat: float = 0.8
    Kd_lat: float = 0.0
    I_lat_max_deg: float = 4.0

    # Derivative measurement LPF time constant (s). Only used if Kd != 0.
    d_lpf_tau_s: float = 0.10

    # Correction magnitude caps (before the absolute pitch/heading clips)
    max_corr_pitch_deg:   float = 20.0
    max_corr_heading_deg: float = 12.0

    # ── Command shaping (outer loop runs every tick, so these are rates) ──
    cmd_lpf_tau_s:        float = 0.08   # ~ old alpha=0.55 @ 20 Hz
    cmd_slew_pitch_dps:   float = 40.0   # ~ old 2.0 deg / 20 Hz step
    cmd_slew_heading_dps: float = 20.0   # ~ old 1.0 deg / 20 Hz step

    # ── DIVE entry ──
    entry_blend_time:     float = 1.0    # s, entry pitch -> dive_pitch_target
    entry_ramp_time:      float = 0.5    # s, PID authority 0 -> 1
    entry_no_pullup_time: float = 0.3    # s, cmd may not be shallower than target

    # ── Heading reference ──
    #   'relative': heading_cmd = current_heading + corr (corr acts as a yaw
    #               rate command through the inner-loop rudder P; recommended)
    #   'locked'  : heading_cmd = heading at DIVE entry + corr
    heading_ref_mode: str = 'relative'

    # Non-zero gimbal lock pitch: a lateral image error then maps to heading
    # through cos(gimbal_pitch). Off by default because main.py locks at 0.
    heading_gain_gimbal_comp: bool = False

    # ── Image validity / robustness ──
    hold_on_lost: bool = True         # hold last error on YOLO dropout
    lost_hold_max_ticks: int = 50     # then decay authority to zero
    lost_decay_ticks:    int = 25
    edge_margin_frac: float = 0.04    # bbox near screen edge -> down-weight
    edge_weight:      float = 0.3

    # Small-bbox noise guard: at long range the bbox is a few px and jitter is
    # amplified. Authority scales linearly over [s_min, s_full] but never
    # below w_far_min (this is noise handling, not the old beta schedule).
    small_bbox_guard: bool = True
    s_min:  float = 0.00005
    s_full: float = 0.0006
    w_far_min: float = 0.45
    s_ema_alpha: float = 0.3

    # Lost target: hold the nominal dive attitude (False, recommended) or fall
    # back to pre_dive_control, which pulls the nose up (True, old behavior).
    lost_fallback_predive: bool = False

    # ── Target KF (still used for the gimbal / trigger distance / logging) ──
    kf_process_noise_accel:    float = 2.5
    kf_measurement_noise_pos:  float = 5.0
    kf_init_pos_std:           float = 30.0
    kf_init_vel_std:           float = 15.0
    velocity_max_speed_mps:    float = 40.0
    kf_dive_reset_pos_std_m:   float = 30.0
    kf_dive_reset_vel_std_mps: float = 15.0
    drift_ema_alpha: float = 0.8
    Z_min: float = 5.0
    Z_ema_alpha: float = 0.3
    warn_if_no_world_state_updates: bool = True

    # Debug
    print_interval_ticks: int = 50


# ══════════════════════════════════════════════════════════════
#  PID with conditional-integration anti-windup
# ══════════════════════════════════════════════════════════════

class _PIDAxis:
    """
    Scalar PID. Derivative acts on the filtered measurement (not the error) to
    avoid setpoint kick; the integrator only accumulates when it is allowed to
    (target visible, output not saturated in the same direction).
    """

    def __init__(self, kp, ki, kd, i_max, d_tau=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_max = i_max
        self.d_tau = d_tau
        self.reset()

    def reset(self):
        self.integral = 0.0
        self._e_filt = 0.0
        self._e_filt_init = False
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def compute(self, error, dt, allow_integrate=True, out_limit=None):
        """
        Returns (output, saturated). Conditional integration: the integrator
        is held when the output is already clipped in the direction the error
        would push it.
        """
        # Derivative on the filtered measurement
        if self.kd != 0.0:
            if not self._e_filt_init:
                self._e_filt = error
                self._e_filt_init = True
                d = 0.0
            else:
                a = math.exp(-dt / max(self.d_tau, 1e-3))
                e_prev = self._e_filt
                self._e_filt = a * self._e_filt + (1.0 - a) * error
                d = (self._e_filt - e_prev) / max(dt, 1e-4)
        else:
            self._e_filt = error
            self._e_filt_init = True
            d = 0.0

        self.p_term = self.kp * error
        self.d_term = self.kd * d

        # Tentative integration
        if allow_integrate and self.ki != 0.0:
            integral_try = self.integral + error * dt
            integral_try = float(np.clip(
                integral_try, -self.i_max / max(self.ki, 1e-9),
                self.i_max / max(self.ki, 1e-9)))
        else:
            integral_try = self.integral

        out_try = self.p_term + self.ki * integral_try + self.d_term

        saturated = False
        if out_limit is not None and abs(out_try) > out_limit:
            saturated = True
            # Only commit the integration if it pushes back toward the limit
            out_prev = self.p_term + self.ki * self.integral + self.d_term
            if abs(out_try) > abs(out_prev):
                integral_try = self.integral      # reject

        self.integral = integral_try
        self.i_term = self.ki * self.integral
        out = self.p_term + self.i_term + self.d_term
        if out_limit is not None:
            out = float(np.clip(out, -out_limit, out_limit))
        return out, saturated

    def decay_integral(self, factor):
        self.integral *= factor
        self.i_term = self.ki * self.integral


# ══════════════════════════════════════════════════════════════
#  Controller
# ══════════════════════════════════════════════════════════════

class PIDDiveController(GimbalIntegratedDiveController):
    """
    Image-error PID dive controller. Public API is identical to
    MPCDiveController so the main loop only needs the import and the
    constructor swapped:

      - control(state, ...)
      - update_target_world_state(...)
      - update_measured_image(...) / get_tvs_debug()
      - notify_dive_started(...) / notify_gimbal_locked(...)
      - update_dkf_drift(...)
      - reset_controllers()
      - .target_kf
    """

    def __init__(self, *args,
                 pid_params=None,
                 ray_estimator=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if pid_params is None:
            pid_params = DivePIDParams()
        self.pid_params = pid_params
        # Alias so existing code that pokes at .mpc_params keeps working
        self.mpc_params = pid_params

        if ray_estimator is None:
            raise ValueError("[PIDDiveController] ray_estimator is required")
        self.ray_estimator = ray_estimator

        p = pid_params

        # ── Outer-loop PIDs ──
        self.pid_lon = _PIDAxis(p.Kp_lon, p.Ki_lon, p.Kd_lon,
                                p.I_lon_max_deg, p.d_lpf_tau_s)
        self.pid_lat = _PIDAxis(p.Kp_lat, p.Ki_lat, p.Kd_lat,
                                p.I_lat_max_deg, p.d_lpf_tau_s)

        # ── Support components (not in the control law) ──
        self.depth_ema = _DepthEMA(alpha=p.Z_ema_alpha, min_value=p.Z_min)
        self.drift_estimator = DriftEstimator(
            c_lon=0.025, c_lat=0.033, alpha=p.drift_ema_alpha)
        self.target_kf = TargetKalmanFilter(
            process_noise_accel=p.kf_process_noise_accel,
            measurement_noise_pos=p.kf_measurement_noise_pos,
            init_pos_std=p.kf_init_pos_std,
            init_vel_std=p.kf_init_vel_std,
            max_speed_mps=p.velocity_max_speed_mps)

        # ── Aircraft / environment cache ──
        self._latest_state = None
        self._latest_gimbal_pitch = 0.0
        self._latest_gimbal_yaw = 0.0
        self._last_ac_x_ned = 0.0
        self._last_ac_y_ned = 0.0
        self._last_ac_vx_ned = 0.0
        self._last_ac_vy_ned = 0.0
        self._last_ac_heading_rad = 0.0
        self._last_ac_altitude = None
        self._last_slant_range = None
        self._last_ground_range = None
        self._last_ray_world = None
        self._last_ray_timestamp = None
        self._world_state_update_count = 0

        # ── Locked at DIVE entry ──
        self._dive_start_time = None
        self._entry_pitch_deg = None
        self._heading_ref_locked = None
        self.gimbal_pitch_locked = 0.0
        self.gimbal_yaw_locked = 0.0
        self.R_b2c_locked = compute_R_b2c(0.0, 0.0)

        # ── Outer-loop command state (LPF + slew memory) ──
        self._cmd_pitch = None            # last issued pitch_cmd (deg)
        self._cmd_heading_corr = 0.0      # last issued heading offset (deg)
        self._last_pitch_cmd_for_log = self.dive_angle
        self._last_yaw_cmd_for_log = 0.0
        self._pid_tick_count = 0

        # ── Image (TVS) state ──
        self._img_u_norm = 0.0
        self._img_v_norm = 0.0
        self._img_s_raw = 0.0
        self._img_s_ema = 0.0
        self._img_detected = False
        self._img_edge_flagged = False
        self._img_lost_ticks = 0
        self._img_update_count = 0
        self._img_w_far = 1.0             # small-bbox noise scale
        self._img_valid = False
        self._w_total = 0.0               # final authority weight [0,1]
        self._pid_active = False
        self._dbg_pitch_corr = 0.0        # applied pitch correction (deg)

        # ── DKF drift (kept for signature compatibility, unused) ──
        self._dkf_drift_lon = 0.0
        self._dkf_drift_lat = 0.0
        self._dkf_drift_valid = False

        # ── KF reset-on-DIVE ──
        self._kf_rejected_count = 0
        self._kf_dive_reset_pending = False
        self._kf_dive_reset_done_count = 0

        # Shared DIVE inner loop: tune inner_loop_controller.py.
        init_dive_inner_loop(self)

        # ── Manual inner-loop test mode ──
        self.manual_mode_enabled = False
        self.manual_dive_pitch_deg = -30.0
        self.manual_trigger_alt_agl = None
        self._manual_initial_heading = None
        self._manual_initial_alt = None
        self._manual_dive_armed = False

        self._print_banner()

    def _print_banner(self):
        p = self.pid_params
        print(f"\n[PIDDiveController] loaded")
        print(f"  arch: image-error PID outer loop + V11.4 cascade inner loop")
        print(f"  Camera: fx={self.ray_estimator.fx:.1f} fy={self.ray_estimator.fy:.1f}, "
              f"screen={self.ray_estimator.screen_width}x{self.ray_estimator.screen_height}")
        print(f"  outer rate: every tick (dt={self._dt:.3f}s, "
              f"~{1.0/max(self._dt,1e-3):.0f} Hz)")
        print(f"  lon PID: Kp={p.Kp_lon} Ki={p.Ki_lon} Kd={p.Kd_lon} "
              f"(I clamp {p.I_lon_max_deg} deg)")
        print(f"  lat PID: Kp={p.Kp_lat} Ki={p.Ki_lat} Kd={p.Kd_lat} "
              f"(I clamp {p.I_lat_max_deg} deg)")
        print(f"  pitch base: {p.dive_pitch_target_deg:+.1f}, "
              f"range [{p.pitch_cmd_min_deg}, {p.pitch_cmd_max_deg}]")
        print(f"  corr caps: pitch={p.max_corr_pitch_deg} "
              f"heading={p.max_corr_heading_deg}, "
              f"hdg max dev={p.heading_cmd_max_dev_deg}")
        print(f"  cmd shaping: LPF tau={p.cmd_lpf_tau_s}s, "
              f"slew pitch={p.cmd_slew_pitch_dps}deg/s heading={p.cmd_slew_heading_dps}deg/s")
        print(f"  entry: blend={p.entry_blend_time}s ramp={p.entry_ramp_time}s "
              f"no-pullup={p.entry_no_pullup_time}s")
        print(f"  heading ref: {p.heading_ref_mode}")
        print(f"  lost target: hold={p.hold_on_lost} "
              f"(max {p.lost_hold_max_ticks} ticks, decay {p.lost_decay_ticks}), "
              f"predive fallback={p.lost_fallback_predive}")
        print(f"  small-bbox guard: {'ON' if p.small_bbox_guard else 'OFF'} "
              f"(s {p.s_min}->{p.s_full}, floor {p.w_far_min})")
        print(f"  rudder slew: {self.rudder_slew_v22}/tick, "
              f"aileron LPF alpha={self.aileron_lpf_alpha}")

    # ════════════════════════════════════════════
    #  Public API
    # ════════════════════════════════════════════

    def control(self, state, target_detected, lateral_offset, longitudinal_offset,
                x_position, current_time,
                gimbal_pitch=0.0, gimbal_yaw=0.0, target_centered=False,
                estimated_distance=None, distance_confidence=0.0,
                y_position=0.0):
        """Cache state/gimbal, then dispatch to the base state machine."""
        self._latest_state = state
        self._latest_gimbal_pitch = gimbal_pitch
        self._latest_gimbal_yaw = gimbal_yaw

        if self.manual_mode_enabled:
            return self._manual_test_control(state, current_time)

        if estimated_distance is not None and distance_confidence > 0.3:
            self.depth_ema.update(estimated_distance)

        return super().control(
            state, target_detected, lateral_offset, longitudinal_offset,
            x_position, current_time,
            gimbal_pitch=gimbal_pitch, gimbal_yaw=gimbal_yaw,
            target_centered=target_centered,
            estimated_distance=estimated_distance,
            distance_confidence=distance_confidence,
            y_position=y_position)

    def update_target_world_state(self, ray_world, slant_range, ground_range,
                                   ac_x_ned, ac_y_ned, ac_vx_ned, ac_vy_ned,
                                   altitude_agl, aircraft_heading_rad, timestamp):
        """
        Feed the target KF and cache aircraft NED state. Call every tick.
        The KF is no longer in the control law; it still drives the gimbal,
        the trigger distance and the logs.
        """
        if ray_world is None or slant_range is None:
            return

        self._last_ac_x_ned = ac_x_ned
        self._last_ac_y_ned = ac_y_ned
        self._last_ac_vx_ned = ac_vx_ned
        self._last_ac_vy_ned = ac_vy_ned
        self._last_ac_heading_rad = aircraft_heading_rad
        self._last_ac_altitude = altitude_agl
        self._last_slant_range = slant_range
        self._last_ground_range = ground_range
        self._last_ray_world = (float(ray_world[0]), float(ray_world[1]),
                                float(ray_world[2]))
        self._last_ray_timestamp = timestamp
        self._world_state_update_count += 1
        self.depth_ema.update(slant_range)

        # DIVE-entry KF reset (position + velocity)
        if self._kf_dive_reset_pending and self.target_kf._initialized:
            meas_x = ac_x_ned + slant_range * float(ray_world[0])
            meas_y = ac_y_ned + slant_range * float(ray_world[1])
            if np.isfinite(meas_x) and np.isfinite(meas_y):
                p = self.pid_params
                old_x = float(self.target_kf.x[0])
                old_y = float(self.target_kf.x[1])
                self.target_kf.x[0] = meas_x
                self.target_kf.x[1] = meas_y
                self.target_kf.x[2] = 0.0
                self.target_kf.x[3] = 0.0
                self.target_kf.P[0, 0] = p.kf_dive_reset_pos_std_m ** 2
                self.target_kf.P[1, 1] = p.kf_dive_reset_pos_std_m ** 2
                self.target_kf.P[2, 2] = p.kf_dive_reset_vel_std_mps ** 2
                self.target_kf.P[3, 3] = p.kf_dive_reset_vel_std_mps ** 2
                self.target_kf.P[0, 1] = 0.0
                self.target_kf.P[1, 0] = 0.0
                self.target_kf.P[2, 3] = 0.0
                self.target_kf.P[3, 2] = 0.0
                self.target_kf._last_update_time = timestamp
                self._kf_dive_reset_pending = False
                self._kf_dive_reset_done_count += 1
                shift = math.hypot(meas_x - old_x, meas_y - old_y)
                print(f"  [KF reset] DIVE entry: target ({old_x:.0f},{old_y:.0f}) -> "
                      f"({meas_x:.0f},{meas_y:.0f}), shift={shift:.0f}m")
                return

        self.target_kf.update(ray_world, slant_range, ac_x_ned, ac_y_ned, timestamp)

    def notify_dive_started(self, timestamp=None):
        """PRE_DIVE -> DIVE transition: seed the outer loop, reset the PIDs."""
        if timestamp is None:
            timestamp = time.time()
        p = self.pid_params
        self._dive_start_time = timestamp

        # PRE_DIVE overshoots below dive_target; start the command at the
        # current pitch and blend toward the target so the inner loop never
        # sees a step pull-up at the transition.
        if self._latest_state is not None:
            self._entry_pitch_deg = float(self._latest_state[11])
            msg = (f"entry pitch {self._entry_pitch_deg:+.1f} -> "
                   f"target {p.dive_pitch_target_deg:+.1f} "
                   f"over {p.entry_blend_time:.1f}s")
        else:
            self._entry_pitch_deg = p.dive_pitch_target_deg
            msg = f"dive_target {p.dive_pitch_target_deg:+.1f} (fallback, no state)"

        self._cmd_pitch = self._entry_pitch_deg
        self._cmd_heading_corr = 0.0
        self._heading_ref_locked = (float(self._latest_state[12])
                                    if self._latest_state is not None else None)

        self.pid_lon.reset()
        self.pid_lat.reset()
        self._pid_tick_count = 0

        self._kf_rejected_count = 0
        self._kf_dive_reset_pending = True

        print(f"\n[PID] DIVE entered. Outer loop runs every tick.")
        print(f"      pitch base: {msg}")
        if p.heading_ref_mode == 'locked' and self._heading_ref_locked is not None:
            print(f"      heading ref locked at {self._heading_ref_locked:+.1f}")

        if (p.warn_if_no_world_state_updates
                and self._world_state_update_count < 5):
            print(f"[PID] warning: update_target_world_state() called only "
                  f"{self._world_state_update_count} times")

    def notify_gimbal_locked(self, timestamp=None, gimbal_pitch=0.0, gimbal_yaw=None):
        """Lock R_b2c at gimbal lock (angles from caller; default 0/0)."""
        if timestamp is None:
            timestamp = time.time()
        if gimbal_yaw is None:
            gimbal_yaw = 0.0

        self.gimbal_pitch_locked = float(gimbal_pitch)
        self.gimbal_yaw_locked = float(gimbal_yaw)
        self.R_b2c_locked = compute_R_b2c(self.gimbal_pitch_locked,
                                          self.gimbal_yaw_locked)

        elapsed = timestamp - self._dive_start_time if self._dive_start_time else 0
        print(f"[PID] gimbal locked at "
              f"({self.gimbal_pitch_locked:+.1f}, {self.gimbal_yaw_locked:+.1f}) "
              f"(DIVE +{elapsed:.1f}s)")
        if abs(self.gimbal_pitch_locked) > 5.0 and not self.pid_params.heading_gain_gimbal_comp:
            print(f"[PID] warning: gimbal locked far from 0; consider "
                  f"heading_gain_gimbal_comp=True")

    def update_dkf_drift(self, u_dot_px, v_dot_px,
                          pitch_rate_deg, heading_rate_deg,
                          screen_width=1920, screen_height=1010):
        """DKF drift is not used by the PID; signature kept for the main loop."""
        self._dkf_drift_lon = v_dot_px / max(screen_height / 2.0, 1.0)
        self._dkf_drift_lat = u_dot_px / max(screen_width / 2.0, 1.0)
        self._dkf_drift_valid = True

    # ════════════════════════════════════════════
    #  Image error input
    # ════════════════════════════════════════════

    def update_measured_image(self, cx, cy, bbox_w, bbox_h,
                               screen_w, screen_h, detected, timestamp=None,
                               agl=None):
        """
        Update the image error from the raw bbox every tick. No proximity
        schedule and no error-driven floor: the PID is active for the whole
        DIVE. Only dropout handling, edge de-weighting and the small-bbox
        noise guard shape the authority here.

        agl is accepted for signature compatibility and is unused.
        """
        p = self.pid_params
        self._img_update_count += 1

        if not detected:
            self._img_detected = False
            self._img_lost_ticks += 1
            if not p.hold_on_lost:
                self._img_valid = False
            # u_norm / v_norm are held; dive_control decides the decay
            return

        self._img_detected = True
        self._img_lost_ticks = 0
        self._img_valid = True

        half_w = max(screen_w / 2.0, 1.0)
        half_h = max(screen_h / 2.0, 1.0)
        self._img_u_norm = float(np.clip((cx - half_w) / half_w, -1.0, 1.0))
        self._img_v_norm = float(np.clip((cy - half_h) / half_h, -1.0, 1.0))

        # bbox screen-area fraction (noise proxy only)
        screen_area = max(screen_w * screen_h, 1.0)
        s_raw = float(max(bbox_w, 0.0) * max(bbox_h, 0.0)) / screen_area
        self._img_s_raw = s_raw
        a = p.s_ema_alpha
        if self._img_s_ema <= 0.0:
            self._img_s_ema = s_raw
        else:
            self._img_s_ema = a * s_raw + (1.0 - a) * self._img_s_ema

        # Small-bbox noise guard (never goes below w_far_min)
        if p.small_bbox_guard:
            span = max(p.s_full - p.s_min, 1e-9)
            frac = float(np.clip((self._img_s_ema - p.s_min) / span, 0.0, 1.0))
            self._img_w_far = p.w_far_min + (1.0 - p.w_far_min) * frac
        else:
            self._img_w_far = 1.0

        # Edge check on the bbox edges, not the center
        bw_half = max(bbox_w, 0.0) / 2.0
        bh_half = max(bbox_h, 0.0) / 2.0
        m_x = p.edge_margin_frac * screen_w
        m_y = p.edge_margin_frac * screen_h
        self._img_edge_flagged = bool(
            (cx - bw_half) < m_x or (cx + bw_half) > (screen_w - m_x)
            or (cy - bh_half) < m_y or (cy + bh_half) > (screen_h - m_y))

    # ════════════════════════════════════════════
    #  DIVE control
    # ════════════════════════════════════════════

    def dive_control(self, lateral_error, longitudinal_error,
                     current_pitch, current_roll, current_heading, current_r,
                     current_q, current_p, target_detected,
                     alpha_meas=None, gamma_true=None):
        """
        DIVE: image-error PID -> (pitch_cmd, heading_cmd) -> cascade inner loop.
        Runs every tick.
        """
        p = self.pid_params
        dt = max(self._dt, 1e-3)

        self.update_dynamic_alpha_compensation(
            current_pitch, alpha_meas=alpha_meas, gamma_true=gamma_true)

        # Late init guard (dive_control reached without notify_dive_started)
        if self._dive_start_time is None:
            self._dive_start_time = time.time()
            self._entry_pitch_deg = current_pitch
            self._cmd_pitch = current_pitch
            self._heading_ref_locked = current_heading
            self.pid_lon.reset()
            self.pid_lat.reset()
            print(f"  [PID] dive_control first run, state auto-initialized")

        # Tick-based elapsed time: deterministic and independent of wall clock
        # (faster-than-realtime replay would otherwise skip the entry blend).
        self._pid_tick_count += 1
        elapsed = (self._pid_tick_count - 1) * dt

        # ── 1. Authority weight ──
        #   entry ramp * dropout hold/decay * edge * small-bbox guard
        w_ramp = (1.0 if p.entry_ramp_time <= 0
                  else float(np.clip(elapsed / p.entry_ramp_time, 0.0, 1.0)))

        if self._img_detected:
            w_lost = 1.0
        elif p.hold_on_lost and self._img_lost_ticks <= p.lost_hold_max_ticks:
            w_lost = 1.0
        elif p.hold_on_lost:
            over = self._img_lost_ticks - p.lost_hold_max_ticks
            w_lost = float(np.clip(1.0 - over / max(p.lost_decay_ticks, 1), 0.0, 1.0))
        else:
            w_lost = 0.0

        w_edge = p.edge_weight if self._img_edge_flagged else 1.0
        w = w_ramp * w_lost * w_edge * self._img_w_far
        w = float(np.clip(w, 0.0, 1.0))
        self._w_total = w

        # ── 2. Optional pre-dive fallback when the target is truly gone ──
        if p.lost_fallback_predive and (not target_detected) and w <= 1e-3:
            rudder, elevator, aileron, pitch_pre, yaw_pre = self.pre_dive_control(
                current_pitch, current_roll, current_heading, current_r,
                current_q=current_q, current_p=current_p,
                gimbal_pitch=self._latest_gimbal_pitch)
            rudder = self._apply_rudder_slew(rudder)
            aileron = self._apply_aileron_lpf(aileron)
            self._pid_active = False
            self._last_pitch_cmd_for_log = pitch_pre
            self._last_yaw_cmd_for_log = yaw_pre
            return rudder, elevator, aileron, pitch_pre, yaw_pre

        # ── 3. PID on the image error ──
        e_lon = -self._img_v_norm      # bbox low -> nose down
        e_lat = self._img_u_norm       # bbox right -> heading right

        if p.heading_gain_gimbal_comp:
            c = max(math.cos(math.radians(self.gimbal_pitch_locked)), 0.2)
            e_lat = e_lat / c

        # Integrate only with a fresh, trustworthy measurement
        allow_i = bool(self._img_detected and not self._img_edge_flagged
                       and w_ramp >= 1.0)

        pitch_corr, _sat_lon = self.pid_lon.compute(
            e_lon, dt, allow_integrate=allow_i, out_limit=p.max_corr_pitch_deg)
        heading_corr, _sat_lat = self.pid_lat.compute(
            e_lat, dt, allow_integrate=allow_i, out_limit=p.max_corr_heading_deg)

        # Bleed the integrators out while authority is decaying
        if w < 1e-3:
            self.pid_lon.decay_integral(0.95)
            self.pid_lat.decay_integral(0.95)

        pitch_corr *= w
        heading_corr *= w

        # ── 4. Pitch command ──
        # Base blends entry pitch -> dive target so the PRE_DIVE overshoot is
        # not commanded away as a step.
        if p.entry_blend_time > 0 and elapsed < p.entry_blend_time:
            k = elapsed / p.entry_blend_time
            pitch_base = ((1.0 - k) * self._entry_pitch_deg
                          + k * p.dive_pitch_target_deg)
        else:
            pitch_base = p.dive_pitch_target_deg

        pitch_cmd_raw = pitch_base + pitch_corr
        pitch_cmd_raw = float(np.clip(pitch_cmd_raw,
                                      p.pitch_cmd_min_deg, p.pitch_cmd_max_deg))

        # DIVE-entry anti-pull-up guard
        if elapsed < p.entry_no_pullup_time:
            pitch_cmd_raw = min(pitch_cmd_raw,
                                max(pitch_base, p.dive_pitch_target_deg))

        pitch_cmd = self._shape_command(
            pitch_cmd_raw, self._cmd_pitch, p.cmd_slew_pitch_dps, dt)
        self._cmd_pitch = pitch_cmd
        self._dbg_pitch_corr = pitch_cmd - pitch_base

        # ── 5. Heading command ──
        heading_corr_raw = float(np.clip(heading_corr,
                                         -p.heading_cmd_max_dev_deg,
                                         p.heading_cmd_max_dev_deg))
        heading_corr = self._shape_command(
            heading_corr_raw, self._cmd_heading_corr, p.cmd_slew_heading_dps, dt)
        self._cmd_heading_corr = heading_corr

        if p.heading_ref_mode == 'locked' and self._heading_ref_locked is not None:
            heading_cmd = self._heading_ref_locked + heading_corr
        else:
            heading_cmd = current_heading + heading_corr

        # ── 6. Inner loop ──
        rudder, elevator, aileron = self._v11_inner_loop(
            pitch_cmd=pitch_cmd, heading_cmd=heading_cmd,
            current_pitch=current_pitch, current_roll=current_roll,
            current_heading=current_heading, current_r=current_r,
            current_q=current_q, current_p=current_p)

        self._pid_active = (w > 1e-3)
        self._last_pitch_cmd_for_log = pitch_cmd
        self._last_yaw_cmd_for_log = heading_cmd

        if (p.print_interval_ticks > 0
                and self._pid_tick_count % p.print_interval_ticks == 0):
            print(f"  [PID] t={elapsed:4.1f}s w={w:.2f} "
                  f"u={self._img_u_norm:+.3f} v={self._img_v_norm:+.3f} | "
                  f"pitch {pitch_base:+.1f}{pitch_corr:+.1f}={pitch_cmd:+.1f} "
                  f"(theta={current_pitch:+.1f}, I={self.pid_lon.i_term:+.2f}) | "
                  f"hdg_off={heading_corr:+.2f} (I={self.pid_lat.i_term:+.2f})")

        return rudder, elevator, aileron, pitch_cmd, heading_cmd

    def _shape_command(self, raw, prev, slew_dps, dt):
        """First-order LPF then a hard rate limit, both in per-tick terms."""
        p = self.pid_params
        if prev is None:
            return raw
        if p.cmd_lpf_tau_s > 0:
            a = math.exp(-dt / p.cmd_lpf_tau_s)
            out = a * prev + (1.0 - a) * raw
        else:
            out = raw
        step = slew_dps * dt
        return float(np.clip(out, prev - step, prev + step))

    def get_tvs_debug(self):
        """
        Diagnostics for the main-loop data log. Key names are unchanged from
        the MPC/TVS version so existing logs and plots keep working;
        tvs_beta is now the PID authority weight.
        """
        return {
            'tvs_active':           1 if self._pid_active else 0,
            'tvs_beta':             float(self._w_total),
            'tvs_u_norm':           float(self._img_u_norm),
            'tvs_v_norm':           float(self._img_v_norm),
            'tvs_s_raw':            float(self._img_s_raw),
            'tvs_s_ema':            float(self._img_s_ema),
            'tvs_corr_pitch_deg':   float(self._dbg_pitch_corr),
            'tvs_corr_heading_deg': float(self._cmd_heading_corr),
            'tvs_detected':         1 if self._img_detected else 0,
            'tvs_edge_flagged':     1 if self._img_edge_flagged else 0,
            'tvs_lost_ticks':       int(self._img_lost_ticks),
            'tvs_entered':          1 if self._dive_start_time is not None else 0,
            'tvs_beta_sched':       float(self._img_w_far),
            'tvs_err_floor':        0.0,
            'tvs_v_norm_ema':       float(self.pid_lon._e_filt),
            # PID term breakdown
            'pid_lon_p':            float(self.pid_lon.p_term),
            'pid_lon_i':            float(self.pid_lon.i_term),
            'pid_lon_d':            float(self.pid_lon.d_term),
            'pid_lat_p':            float(self.pid_lat.p_term),
            'pid_lat_i':            float(self.pid_lat.i_term),
            'pid_lat_d':            float(self.pid_lat.d_term),
        }

    # ════════════════════════════════════════════
    #  Manual inner-loop test mode
    # ════════════════════════════════════════════

    def _manual_test_control(self, state, current_time):
        """SEARCH holds altitude; DIVE feeds a fixed pitch_cmd + locked heading."""
        airspeed     = state[0]
        p_rate       = state[7]
        q_rate       = state[8]
        r_rate       = state[9]
        roll         = state[10]
        pitch        = state[11]
        heading      = state[12]
        altitude_msl = state[19]
        altitude_agl = altitude_msl - self._runway_elev

        if self._manual_initial_heading is None:
            self._manual_initial_heading = heading
            self._manual_initial_alt = altitude_agl
            print(f"\n{'='*60}")
            print(f"[MANUAL TEST] started (PID + vision bypassed)")
            print(f"  initial heading = {heading:+.1f} (locked)")
            print(f"  initial AGL     = {altitude_agl:.1f} m")
            print(f"  dive pitch cmd  = {self.manual_dive_pitch_deg:+.1f}")
            if self.manual_trigger_alt_agl is None:
                print(f"  trigger: immediate DIVE")
                self._manual_dive_armed = True
            else:
                print(f"  trigger: AGL < {self.manual_trigger_alt_agl} m")
            print(f"{'='*60}\n")

        if not self._manual_dive_armed:
            if (self.manual_trigger_alt_agl is not None
                    and altitude_agl < self.manual_trigger_alt_agl):
                self._manual_dive_armed = True
                print(f"\n[MANUAL TEST] DIVE @ AGL={altitude_agl:.1f}m, "
                      f"pitch_cmd -> {self.manual_dive_pitch_deg:+.1f}\n")

        if not self._manual_dive_armed:
            rudder, elevator, aileron, pitch_cmd, yaw_cmd = self.search_control(
                pitch, roll, heading, altitude_agl,
                gimbal_yaw_cmd=0.0, current_p=p_rate, current_r=r_rate)
            mode = 'SEARCH'
        else:
            pitch_cmd = self.manual_dive_pitch_deg
            heading_cmd = self._manual_initial_heading
            rudder, elevator, aileron = self._v11_inner_loop(
                pitch_cmd=pitch_cmd, heading_cmd=heading_cmd,
                current_pitch=pitch, current_roll=roll,
                current_heading=heading, current_r=r_rate,
                current_q=q_rate, current_p=p_rate)
            yaw_cmd = heading_cmd
            mode = 'DIVE'

        if mode != self.current_mode:
            print(f"[MANUAL] mode: {self.current_mode} -> {mode}")
            self.current_mode = mode
        self.mode_history.append(mode)
        self.last_pitch_cmd = pitch_cmd
        self.last_yaw_cmd = yaw_cmd
        self._last_pitch_cmd_for_log = pitch_cmd
        self._last_yaw_cmd_for_log = yaw_cmd

        throttle = self.calculate_throttle(altitude_agl, airspeed, mode)
        return elevator, aileron, rudder, throttle, pitch_cmd, yaw_cmd

    # ════════════════════════════════════════════
    #  Cascade inner loop (unchanged from V11.4)
    # ════════════════════════════════════════════

    def _v11_inner_loop(self, pitch_cmd, heading_cmd,
                         current_pitch, current_roll, current_heading,
                         current_r, current_q, current_p):
        return compute_inner_loop(
            self, pitch_cmd, heading_cmd, current_pitch, current_roll,
            current_heading, current_r, current_q, current_p)

    def _apply_rudder_slew(self, rudder):
        return apply_rudder_slew(self, rudder)

    def _apply_aileron_lpf(self, aileron):
        return apply_aileron_lpf(self, aileron)

    # ════════════════════════════════════════════
    #  Reset
    # ════════════════════════════════════════════

    def reset_controllers(self):
        """Reset all PID + base-class state."""
        super().reset_controllers()
        self.pid_lon.reset()
        self.pid_lat.reset()
        self.drift_estimator.reset()
        if hasattr(self.target_kf, 'reset'):
            self.target_kf.reset()
        self.depth_ema.reset()

        self._cmd_pitch = None
        self._cmd_heading_corr = 0.0
        self._pid_tick_count = 0
        self._pid_active = False
        self._w_total = 0.0
        self._dbg_pitch_corr = 0.0

        self._dive_start_time = None
        self._entry_pitch_deg = None
        self._heading_ref_locked = None
        self.gimbal_pitch_locked = 0.0
        self.gimbal_yaw_locked = 0.0
        self.R_b2c_locked = compute_R_b2c(0.0, 0.0)

        self._last_slant_range = None
        self._last_ground_range = None
        self._last_ac_altitude = None
        self._last_ac_x_ned = 0.0
        self._last_ac_y_ned = 0.0
        self._last_ray_world = None
        self._world_state_update_count = 0

        self._dkf_drift_lon = 0.0
        self._dkf_drift_lat = 0.0
        self._dkf_drift_valid = False
        self._kf_rejected_count = 0
        self._kf_dive_reset_pending = False
        self._kf_dive_reset_done_count = 0

        self._prev_rudder_v22 = 0.0
        self._rudder_lpf_state = 0.0
        self._aileron_lpf_state = 0.0

        self._manual_initial_heading = None
        self._manual_initial_alt = None
        self._manual_dive_armed = False

        # Image state
        self._img_u_norm = 0.0
        self._img_v_norm = 0.0
        self._img_s_raw = 0.0
        self._img_s_ema = 0.0
        self._img_detected = False
        self._img_edge_flagged = False
        self._img_lost_ticks = 0
        self._img_update_count = 0
        self._img_w_far = 1.0
        self._img_valid = False

        print("[PID] reset")
