"""
Dive controller: SEARCH/PRE_DIVE/DIVE/TERMINAL state machine,
3DOF point-mass MPC, target KF, and V11.4 cascade inner loop.

MPC internal model: 3DOF point-mass kinematics + first-order command response
+ energy dynamics (gravity + drag).

State (target-relative NED):
  x = [Dx, Dy, h, V, chi, gamma]
  Dx_dot = V cos(gamma) cos(chi)
  Dy_dot = V cos(gamma) sin(chi)
  h_dot  = V sin(gamma)
  V_dot  = -g sin(gamma) - k_drag V^2
  chi_dot   = (chi_cmd - chi) / tau_chi
  gamma_dot = (gamma_cmd - gamma) / tau_gamma
Control: u = [chi_cmd, gamma_cmd] (rad).

Image cost via geometric projection (no IBVS jacobian):
  v_ned = (-Dx, -Dy, h); R_ned2body ~ Ry(gamma + alpha_nom) Rz(chi)
  v_cam = R_b2c v_body; x_n = v_cam_x / v_cam_z, y_n = v_cam_y / v_cam_z

Command flow: MPC outputs (chi_cmd, gamma_cmd) ->
  pitch_cmd_deg   = deg(gamma_cmd) + alpha_nominal_deg
  heading_cmd_deg = current_heading + deg(chi_cmd - chi)
-> V11.4 cascade inner loop -> (rudder, elevator, aileron).

References: Stevens/Lewis/Johnson "Aircraft Control and Simulation" 2.4;
Beard & McLain "Small Unmanned Aircraft" ch. 9.
"""

import numpy as np
from inner_loop_controller import (
    init_dive_inner_loop, compute_inner_loop,
    apply_rudder_slew, apply_aileron_lpf,
)
import math
import time
import casadi as ca
from collections import deque
from dataclasses import dataclass
from utils.PID import PID


# ==================== Constants ====================
RCMQ_RUNWAY_18 = 50 * 0.3048  # 50 ft in meters
RCMQ_RUNWAY_ELEV = 201.79


# ══════════════════════════════════════════════════════════════
#  Rotation helpers (numpy)
# ══════════════════════════════════════════════════════════════

# Gimbal -> Camera (CV), consistent with RayProjectionDistanceEstimator
_R_G2C = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
], dtype=float)

def _Ry_np(deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def _Rz_np(deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

def compute_R_b2c(gimbal_pitch_deg, gimbal_yaw_deg):
    """Body -> Camera rotation: R_b2c = R_g2c @ Ry(gimbal_pitch) @ Rz(gimbal_yaw)."""
    R_b2g = _Ry_np(gimbal_pitch_deg) @ _Rz_np(gimbal_yaw_deg)
    return _R_G2C @ R_b2g


# ══════════════════════════════════════════════════════════════
#  State-machine base controller
# ══════════════════════════════════════════════════════════════

class GimbalIntegratedDiveController:
    """
    Multi-phase dive attack state machine: SEARCH -> PRE_DIVE -> DIVE -> TERMINAL.

    PRE_DIVE commands pitch to the dive angle directly (decoupled from gimbal
    tracking) to avoid gimbal-aircraft coupling oscillation.
    dive_control() is provided by the MPC subclass.
    """

    def __init__(self,
                 gamma=30.0,
                 dt=0.05,
                 tch=RCMQ_RUNWAY_18,
                 runway_elev=RCMQ_RUNWAY_ELEV,
                 des_u=40,
                 trigger_distance=None,   # manual ground trigger distance (m); None = auto
                 distance_margin=1.3,     # trigger margin multiplier (>1 = earlier)
                 min_confidence=0.3,      # min range-estimate confidence to trigger
                 target_position=None,    # (x, y) NED; enables position-based trigger
                 search_target_altitude=None,
                 debug_altitude_hold=True):
        """
        Args:
            gamma: dive angle (deg, positive down).
            tch: pull-up altitude AGL (m).
            trigger_distance: None -> auto = altitude / tan(gamma).
        """
        self._gamma = gamma
        self._h_thresh = tch + runway_elev
        self._runway_elev = runway_elev
        self._des_u = des_u
        self._dt = dt

        # Dive geometry
        self.dive_angle = -abs(gamma)
        self.dive_angle_rad = math.radians(abs(gamma))
        # Overshoot: command deeper than dive_angle so the aircraft pitch
        # sweeps past LoS and the gimbal naturally clips at 0 before DIVE lock.
        self.pre_dive_overshoot = 7.0
        self.pre_dive_pitch_target = self.dive_angle - self.pre_dive_overshoot

        # Dynamic angle-of-attack compensation (disabled: no alpha comp)
        self.alpha_bias = 0.0
        self.use_dynamic_alpha_comp = False
        self.dynamic_alpha_tau = 0.35      # s, LPF time constant
        self.dynamic_alpha_min = 0.3       # deg
        self.dynamic_alpha_max = 8.0       # deg
        self.alpha_sensor_min = -2.0       # deg, outlier filter
        self.alpha_sensor_max = 12.0       # deg
        self.dynamic_alpha_est = self.alpha_bias
        self.dynamic_pitch_base = self.dive_angle - self.alpha_bias

        # Distance trigger
        self.trigger_distance_override = trigger_distance
        self.distance_margin = distance_margin
        self.min_trigger_confidence = min_confidence
        self.target_position = target_position   # position trigger has priority

        # Altitude
        self.terminal_altitude = tch
        self.search_target_altitude = search_target_altitude
        self.search_altitude_initialized = False
        self._frozen_trigger_slant = None
        self._frozen_trigger_ground = None

        # ==================== SEARCH ====================
        self.search_pitch = 0.0
        self.search_heading = 0.0

        # Small I terms remove steady-state altitude/pitch/heading errors
        self.search_altitude_kp = 0.7
        self.search_altitude_ki = 0.05
        self.search_altitude_kd = 0.001
        self.search_altitude_pid = PID(dt, self.search_altitude_kp,
                                       self.search_altitude_ki,
                                       self.search_altitude_kd)

        self.search_pitch_kp = 0.05
        self.search_pitch_ki = 0.02
        self.search_pitch_kd = 0.001
        self.search_pitch_pid = PID(dt, self.search_pitch_kp,
                                    self.search_pitch_ki,
                                    self.search_pitch_kd)

        self.heading_kp = 0.020
        self.heading_ki = 0.005
        self.heading_kd = 0.001
        self.heading_pid = PID(dt, self.heading_kp, self.heading_ki, self.heading_kd)

        # ==================== PRE_DIVE ====================
        self.pre_dive_target_pitch = self.dive_angle
        self.pre_dive_initial_gimbal_pitch = 0.0

        self.pre_dive_pitch_rate_limit = 20.0
        self._pre_dive_ramped_pitch_cmd = None

        # Fast-push gains: command dive_angle directly, elevator slew limits rate
        self.pre_dive_K_theta = 3.0
        self.pre_dive_K_q = 0.06
        self.pre_dive_max_q_cmd = 25.0

        # Elevator slew (the actual rate limiter)
        self.pre_dive_elevator_slew = 0.20

        # Elevator output LPF: kills high-frequency chatter driven by
        # K_q * q_noise without the phase-lag issues of filtering q itself.
        # alpha 0.7 ~ 22 Hz cutoff; balances chatter vs steady-state lag.
        self.elevator_lpf_alpha = 0.7
        self._elevator_lpf_state = 0.0
        self.prev_pre_dive_elevator = 0.0

        # PRE_DIVE state
        self.pre_dive_start_time = None
        self.pre_dive_complete = False
        self.gimbal_locked = False
        self.pre_dive_triggered = False
        # B2 completion hysteresis: condition must hold this long (0 = instant)
        self._b2_first_satisfied_time = None
        self.b2_stable_duration_sec = 0.0

        self.pitch_alignment_tolerance = 3.0
        self.gimbal_zero_tolerance = 5.0
        self.pitch_alignment_hold_time = 0.3
        self.pitch_alignment_start = None

        # ==================== Throttle ====================
        self.throttle_kp = 500.0
        self.throttle_ki = 0.0
        self.throttle_kd = 80.0
        self._u_pid = PID(dt, self.throttle_kp, self.throttle_ki, self.throttle_kd)

        # ==================== Control limits ====================
        self.max_rudder = 0.7
        self.max_elevator = 1.0
        self.max_aileron = 0.15

        self.search_elevator_sign = -1
        self.pre_dive_elevator_sign = -1
        self.dive_elevator_sign = -1

        # ==================== State bookkeeping ====================
        self.current_mode = 'INIT'
        self.mode_history = []
        self.current_x = 0.0

        self.gimbal_target_pitch = 0.0
        self.heading_bias_compensation = 0.0
        self.last_pitch_cmd = 0.0
        self.last_yaw_cmd = 0.0

        # Distance trigger bookkeeping
        self.last_estimated_distance = None
        self.last_trigger_distance = None

        self._last_mode_print_time = 0

        # Body-rate LPFs (per-axis): q is a damping signal in the theta-Q
        # cascade, so filtering it adds phase lag and causes ~10 Hz
        # self-excited oscillation -> q LPF off. r/p LPFs stay on.
        self.rate_lpf_alpha_q = 1.0     # 1.0 = off
        self.rate_lpf_alpha_r = 0.3     # ~6.8 Hz cutoff
        self.rate_lpf_alpha_p = 0.3
        self._q_filt = 0.0
        self._r_filt = 0.0
        self._p_filt = 0.0
        self.rudder_slew_v14 = 0.015    # per 100 Hz tick
        self._prev_rudder_v14 = 0.0

        print("=" * 70)
        print("Gimbal Integrated Dive Controller")
        print("=" * 70)
        print(f"  dive angle: {self.dive_angle} deg")
        if trigger_distance is not None:
            print(f"  trigger ground distance (manual): {trigger_distance:.1f}m")
        else:
            print(f"  trigger ground distance: auto = altitude / tan({abs(gamma)})")
        print(f"  distance margin: {distance_margin:.1f}x")
        if self.target_position is not None:
            print(f"  position trigger: target=({self.target_position[0]:.0f}, {self.target_position[1]:.0f})")
        else:
            print(f"  trigger method: ray-projection range")
        print(f"  min confidence: {min_confidence}")
        print(f"  pull-up altitude: {self.terminal_altitude:.2f}m AGL")
        print("=" * 70)

    # ==================== Trigger distance ====================

    def compute_trigger_ground_range(self, altitude_agl):
        """Trigger ground range: manual override or altitude / tan(gamma)."""
        if self.trigger_distance_override is not None:
            return self.trigger_distance_override * self.distance_margin

        if self.dive_angle_rad > 0.01:
            base_range = altitude_agl / math.tan(self.dive_angle_rad)
        else:
            base_range = 99999.0

        return base_range * self.distance_margin

    def compute_trigger_slant_range(self, altitude_agl):
        """Trigger slant range = altitude / sin(gamma) (compared to ray estimate)."""
        if self.trigger_distance_override is not None:
            ground_range = self.trigger_distance_override * self.distance_margin
            return math.sqrt(ground_range ** 2 + altitude_agl ** 2)

        if self.dive_angle_rad > 0.01:
            base_slant = altitude_agl / math.sin(self.dive_angle_rad)
        else:
            base_slant = 99999.0

        return base_slant * self.distance_margin

    def update_dynamic_alpha_compensation(self, current_pitch, alpha_meas=None, gamma_true=None):
        """Update dynamic AoA compensation (pass-through when disabled)."""
        if not self.use_dynamic_alpha_comp:
            self.dynamic_alpha_est = self.alpha_bias
            self.dynamic_pitch_base = self.dive_angle - self.alpha_bias
            self.pre_dive_pitch_target = self.dynamic_pitch_base - self.pre_dive_overshoot
            return self.dynamic_pitch_base

        candidates = []

        if alpha_meas is not None and np.isfinite(alpha_meas):
            if self.alpha_sensor_min <= alpha_meas <= self.alpha_sensor_max:
                candidates.append(float(alpha_meas))

        if gamma_true is not None and np.isfinite(gamma_true):
            inferred_alpha = float(current_pitch - gamma_true)
            if self.alpha_sensor_min <= inferred_alpha <= self.alpha_sensor_max:
                candidates.append(inferred_alpha)

        if candidates:
            raw_alpha = float(np.mean(candidates))
        else:
            raw_alpha = self.dynamic_alpha_est if np.isfinite(self.dynamic_alpha_est) else self.alpha_bias

        lpf_gain = self._dt / (self.dynamic_alpha_tau + self._dt)
        self.dynamic_alpha_est = self.dynamic_alpha_est + lpf_gain * (raw_alpha - self.dynamic_alpha_est)
        self.dynamic_alpha_est = float(np.clip(self.dynamic_alpha_est,
                                               self.dynamic_alpha_min,
                                               self.dynamic_alpha_max))

        self.dynamic_pitch_base = self.dive_angle - self.dynamic_alpha_est
        self.pre_dive_pitch_target = self.dynamic_pitch_base - self.pre_dive_overshoot
        return self.dynamic_pitch_base

    def get_current_dive_pitch_base(self):
        """Current dive base pitch (includes dynamic alpha comp if enabled)."""
        if self.use_dynamic_alpha_comp:
            return self.dynamic_pitch_base
        return self.dive_angle - self.alpha_bias

    # ==================== Mode determination ====================

    def determine_mode(self, altitude_agl, x_position, current_pitch,
                      target_detected, target_centered, gimbal_pitch, current_time,
                      estimated_distance=None, distance_confidence=0.0,
                      y_position=0.0):
        """Decide control mode for this tick."""
        self.current_x = x_position

        # Terminal check
        if altitude_agl < self.terminal_altitude:
            return 'TERMINAL'

        # PRE_DIVE done -> DIVE
        if self.pre_dive_complete:
            return 'DIVE'

        # PRE_DIVE triggered stays triggered (no fallback to SEARCH)
        if self.pre_dive_triggered:
            # Timeout safety: force DIVE after 3 s
            if self.pre_dive_start_time is not None:
                pre_dive_elapsed = current_time - self.pre_dive_start_time
                if pre_dive_elapsed > 3.0:
                    self.pre_dive_complete = True
                    self.gimbal_locked = False
                    print(f"\n[PRE_DIVE] timeout -> DIVE ({pre_dive_elapsed:.1f}s)")
                    print(f"  pitch: {current_pitch:.1f} | gimbal: {gimbal_pitch:.1f}")
                    return 'DIVE'

            # Pitch reached target (1 deg early-trigger tolerance)
            pitch_reached = current_pitch <= self.pre_dive_pitch_target + 1.0

            # B2 image-based completion: gimbal swept to ~0 + target centered.
            # Late PRE_DIVE the aircraft pitch passes LoS and the gimbal clips
            # at 0; entering DIVE at that moment keeps the target centered at
            # gimbal lock instead of drifting up in the image.
            gimbal_settled = abs(gimbal_pitch) < 3.0
            b2_instant = gimbal_settled and target_centered

            # Hysteresis: condition must hold b2_stable_duration_sec
            if b2_instant:
                if self._b2_first_satisfied_time is None:
                    self._b2_first_satisfied_time = current_time
                    print(f"  [PRE_DIVE] B2 condition met, hold timer "
                          f"{self.b2_stable_duration_sec}s "
                          f"(pitch={current_pitch:.1f}, gimbal={gimbal_pitch:.2f})")
                b2_held_duration = current_time - self._b2_first_satisfied_time
                b2_completed = b2_held_duration >= self.b2_stable_duration_sec
            else:
                if self._b2_first_satisfied_time is not None:
                    print(f"  [PRE_DIVE] B2 condition broken, reset timer "
                          f"(gimbal_settled={gimbal_settled}, target_centered={target_centered})")
                self._b2_first_satisfied_time = None
                b2_completed = False

            if b2_completed or pitch_reached:
                self.pre_dive_complete = True
                self.gimbal_locked = False
                line_of_sight = current_pitch + gimbal_pitch
                if b2_completed and not pitch_reached:
                    held = current_time - self._b2_first_satisfied_time
                    completion_reason = f"B2 stable hold {held:.2f}s (target centered + gimbal=0)"
                else:
                    completion_reason = "pitch reached"
                print(f"\n[PRE_DIVE] -> DIVE, reason: {completion_reason}")
                print(f"  pitch: {current_pitch:.1f} (target {self.pre_dive_pitch_target:.1f})")
                print(f"  gimbal pitch: {gimbal_pitch:.1f}  target_centered: {target_centered}")
                print(f"  LoS: {line_of_sight:.1f}")
                return 'DIVE'

            return 'PRE_DIVE'

        # ========== SEARCH -> PRE_DIVE trigger ==========
        trigger_by_distance = False
        trigger_method = ""

        # Show the frozen trigger distance
        if hasattr(self, '_frozen_trigger_slant') and self._frozen_trigger_slant is not None:
            self.last_trigger_distance = self._frozen_trigger_slant

        # Position trigger (priority)
        if self.target_position is not None and target_detected:
            tx, ty = self.target_position
            ground_dist = math.sqrt((x_position - tx)**2 + (y_position - ty)**2)
            trigger_ground = self.compute_trigger_ground_range(altitude_agl)

            if ground_dist <= trigger_ground:
                trigger_by_distance = True
                trigger_method = "position"
                self.last_estimated_distance = math.sqrt(ground_dist**2 + altitude_agl**2)

        # Ray-projection trigger (fallback when no target_position)
        if not trigger_by_distance and self.target_position is None:
            if (target_detected and
                    estimated_distance is not None and
                    distance_confidence >= self.min_trigger_confidence):

                trigger_slant = self.last_trigger_distance or self.compute_trigger_slant_range(altitude_agl)
                self.last_estimated_distance = estimated_distance
                trigger_by_distance = (estimated_distance <= trigger_slant)
                trigger_method = "ray_projection"

        if trigger_by_distance:
            self.pre_dive_triggered = True
            self.pre_dive_start_time = current_time
            self.pre_dive_initial_gimbal_pitch = gimbal_pitch
            self.pre_dive_pitch_target = self.dynamic_pitch_base - self.pre_dive_overshoot

            trigger_ground = self.compute_trigger_ground_range(altitude_agl)
            if self.last_estimated_distance and self.last_estimated_distance > altitude_agl:
                est_ground = math.sqrt(self.last_estimated_distance ** 2 - altitude_agl ** 2)
            else:
                est_ground = 0.0

            print(f"\n[PRE_DIVE] start!")
            print(f"  trigger: {trigger_method}")
            if trigger_method == "position":
                tx, ty = self.target_position
                gnd = math.sqrt((x_position - tx)**2 + (y_position - ty)**2)
                print(f"  aircraft: ({x_position:.0f}, {y_position:.0f})")
                print(f"  target: ({tx:.0f}, {ty:.0f})")
                print(f"  ground dist: {gnd:.0f}m <= trigger {trigger_ground:.0f}m")
            else:
                print(f"  est slant: {estimated_distance:.1f}m (conf: {distance_confidence:.2f})")
                print(f"  est ground: {est_ground:.1f}m")
                print(f"  trigger ground: {trigger_ground:.1f}m")
            print(f"  altitude: {altitude_agl:.1f}m AGL")
            print(f"  initial gimbal pitch: {gimbal_pitch:.1f}")
            print(f"  aircraft pitch target: {self.pre_dive_pitch_target:.1f}")

            return 'PRE_DIVE'

        return 'SEARCH'

    # ==================== SEARCH control ====================

    def search_control(self, current_pitch, current_roll, current_heading, altitude_agl,
                      gimbal_yaw_cmd, current_p=0.0, current_r=0.0):
        """Search mode: hold altitude, level flight."""
        if not self.search_altitude_initialized:
            if self.search_target_altitude is None:
                self.search_target_altitude = altitude_agl
            self.search_altitude_initialized = True

            # Freeze trigger distances here (do not track altitude afterwards)
            self._frozen_trigger_slant = self.compute_trigger_slant_range(self.search_target_altitude)
            self._frozen_trigger_ground = self.compute_trigger_ground_range(self.search_target_altitude)

            print(f"[SEARCH] altitude hold: {self.search_target_altitude:.1f}m AGL")
            print(f"  frozen trigger: slant={self._frozen_trigger_slant:.0f}m, "
                  f"ground={self._frozen_trigger_ground:.0f}m")

        altitude_error = self.search_target_altitude - altitude_agl
        pitch_adjustment = self.search_altitude_pid(altitude_error)
        pitch_adjustment = np.clip(pitch_adjustment, -6.0, 6.0)

        pitch_cmd = self.search_pitch + pitch_adjustment
        pitch_cmd = np.clip(pitch_cmd, -15.0, 5.0)

        pitch_error = current_pitch - pitch_cmd
        elevator = self.search_elevator_sign * self.search_pitch_pid(pitch_error)
        elevator = np.clip(elevator, -self.max_elevator, self.max_elevator)

        yaw_cmd = self.search_heading + self.heading_bias_compensation

        # Large gimbal yaw -> steer aircraft toward it
        gimbal_yaw_threshold = 20.0
        if abs(gimbal_yaw_cmd) > gimbal_yaw_threshold:
            yaw_cmd += gimbal_yaw_cmd * 0.3

        heading_error = yaw_cmd - current_heading
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        rudder = self.heading_pid(heading_error)
        rudder = np.clip(rudder, -self.max_rudder, self.max_rudder)

        aileron = -0.05 * current_roll - 0.015 * current_p
        aileron = np.clip(aileron, -0.25, 0.25)

        return rudder, elevator, aileron, pitch_cmd, yaw_cmd

    # ==================== PRE_DIVE control ====================

    def pre_dive_control(self, current_pitch, current_roll, current_heading, current_r,
                         current_q=0.0, current_p=0.0, gimbal_pitch=0.0):
        """
        Pre-dive: command dive pitch directly (no ramp), elevator slew limits
        the rate. Short PRE_DIVE (<0.5 s) keeps roll coupling from accumulating.
        """
        # Pitch command: dive base - overshoot
        pitch_cmd = self.pre_dive_pitch_target
        pitch_cmd = np.clip(pitch_cmd, -60.0, 5.0)
        self.pre_dive_target_pitch = pitch_cmd

        # Dual-loop theta-Q
        pitch_error = pitch_cmd - current_pitch
        q_cmd = self.pre_dive_K_theta * pitch_error
        q_cmd = np.clip(q_cmd, -self.pre_dive_max_q_cmd, self.pre_dive_max_q_cmd)

        q_error = q_cmd - current_q
        elevator = self.pre_dive_K_q * q_error

        # Elevator slew (the actual rate limiter)
        max_step = self.pre_dive_elevator_slew
        elevator = np.clip(elevator,
                          self.prev_pre_dive_elevator - max_step,
                          self.prev_pre_dive_elevator + max_step)
        elevator = np.clip(elevator, -1.0, 1.0)

        # Elevator output LPF (after slew)
        a = self.elevator_lpf_alpha
        elevator = a * elevator + (1.0 - a) * self._elevator_lpf_state
        self._elevator_lpf_state = elevator

        self.prev_pre_dive_elevator = elevator

        # Heading: yaw-rate damping + light heading hold
        yaw_cmd = self.heading_bias_compensation
        heading_error = yaw_cmd - current_heading
        if heading_error > 180: heading_error -= 360
        elif heading_error < -180: heading_error += 360
        rudder = 0.01 * heading_error - 0.06 * current_r
        rudder = np.clip(rudder, -0.15, 0.15)

        # Roll: rate damping
        aileron = -0.05 * current_roll - 0.015 * current_p
        aileron = np.clip(aileron, -0.25, 0.25)

        return rudder, elevator, aileron, pitch_cmd, yaw_cmd

    # ==================== TERMINAL control ====================

    def terminal_control(self, current_pitch, current_heading):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # ==================== Throttle ====================

    def calculate_throttle(self, altitude_agl, airspeed, mode):
        airspeed_ms = airspeed * 0.5144

        if mode == 'SEARCH':
            target_u = 30
            err_u = target_u - airspeed_ms
            fu = self._u_pid(err_u) + 2000
            throttle = fu / 5000.0
            throttle = np.clip(throttle, -0.5, 0.8)

        elif mode == 'PRE_DIVE':
            target_u = 30
            err_u = target_u - airspeed_ms
            throttle_base = 0.3
            throttle_adjust = 0.01 * err_u
            throttle = throttle_base + throttle_adjust
            throttle = np.clip(throttle, 0.1, 0.5)
            if airspeed_ms > 55.0:
                throttle = 0.1

        elif mode == 'DIVE':
            min_dive_speed = 25.0
            max_dive_speed = 80.0
            if airspeed_ms < min_dive_speed:
                throttle = 0.2
            elif airspeed_ms > max_dive_speed:
                throttle = -1.0
            else:
                throttle = -0.8

        else:
            throttle = -0.5

        if airspeed > 210:
            throttle = -1.0

        return throttle

    # ==================== Main control ====================

    def control(self, state, target_detected, lateral_offset, longitudinal_offset,
               x_position, current_time,
               gimbal_pitch=0.0, gimbal_yaw=0.0, target_centered=False,
               estimated_distance=None, distance_confidence=0.0,
               y_position=0.0):
        """
        Main control entry.

        Returns:
            (elevator, aileron, rudder, throttle, pitch_cmd, yaw_cmd)
        """
        airspeed = state[0]
        phi = state[10]
        theta = state[11]
        psi = state[12]
        altitude_msl = state[19]
        altitude_agl = altitude_msl - self._runway_elev
        alpha = state[13] if len(state) > 13 else None
        gamma_true = state[15] if len(state) > 15 else None

        # Per-axis body-rate LPF (q off; see attribute comment)
        a_q = self.rate_lpf_alpha_q
        a_r = self.rate_lpf_alpha_r
        a_p = self.rate_lpf_alpha_p
        self._p_filt = a_p * state[7] + (1.0 - a_p) * self._p_filt
        self._q_filt = a_q * state[8] + (1.0 - a_q) * self._q_filt
        self._r_filt = a_r * state[9] + (1.0 - a_r) * self._r_filt
        p = self._p_filt
        q = self._q_filt
        r = self._r_filt

        # Update dynamic alpha comp first so PRE_DIVE/DIVE use fresh base
        self.update_dynamic_alpha_compensation(theta, alpha_meas=alpha, gamma_true=gamma_true)

        mode = self.determine_mode(
            altitude_agl, x_position, theta,
            target_detected, target_centered, gimbal_pitch, current_time,
            estimated_distance=estimated_distance,
            distance_confidence=distance_confidence,
            y_position=y_position
        )

        if mode != self.current_mode:
            print(f"\n[Controller] mode switch: {self.current_mode} -> {mode}")
            print(f"  x: {x_position:.1f}m | Alt AGL: {altitude_agl:.1f}m")
            print(f"  pitch: {theta:.1f} | gimbal pitch: {gimbal_pitch:.1f}")
            if estimated_distance is not None:
                print(f"  est dist: {estimated_distance:.1f}m (conf: {distance_confidence:.2f})")
            self.current_mode = mode

        self.mode_history.append(mode)

        if mode == 'PRE_DIVE':
            self.gimbal_target_pitch = theta + gimbal_pitch

        if mode == 'SEARCH':
            rudder, elevator, aileron, pitch_cmd, yaw_cmd = self.search_control(
                theta, phi, psi, altitude_agl, gimbal_yaw, current_p=p, current_r=r)

        elif mode == 'PRE_DIVE':
            rudder, elevator, aileron, pitch_cmd, yaw_cmd = self.pre_dive_control(
                theta, phi, psi, r, current_q=q, current_p=p, gimbal_pitch=gimbal_pitch)

        elif mode == 'DIVE':
            rudder, elevator, aileron, pitch_cmd, yaw_cmd = self.dive_control(
                lateral_offset, longitudinal_offset, theta, phi, psi, r,
                q, p, target_detected, alpha_meas=alpha, gamma_true=gamma_true)

        else:
            rudder, elevator, aileron, pitch_cmd, yaw_cmd = self.terminal_control(theta, psi)

        self.last_pitch_cmd = pitch_cmd
        self.last_yaw_cmd = yaw_cmd

        throttle = self.calculate_throttle(altitude_agl, airspeed, mode)

        return elevator, aileron, rudder, throttle, pitch_cmd, yaw_cmd

    # ==================== Helpers ====================

    def get_gimbal_target_pitch(self):
        return self.gimbal_target_pitch

    def is_in_predive(self):
        return self.current_mode == 'PRE_DIVE'

    def is_gimbal_should_lock(self):
        return self.gimbal_locked

    def set_trigger_distance(self, distance):
        self.trigger_distance_override = distance
        print(f"[Controller] trigger ground distance: {distance:.1f}m")

    def set_heading_bias_compensation(self, bias):
        self.heading_bias_compensation = bias
        print(f"[Controller] heading bias compensation: {bias:+.1f}")

    def reset_controllers(self):
        """Reset all controller state."""
        self.search_altitude_pid._integral = 0.0
        self.search_altitude_pid._prev_error = 0.0
        self.search_pitch_pid._integral = 0.0
        self.search_pitch_pid._prev_error = 0.0
        self.heading_pid._integral = 0.0
        self.heading_pid._prev_error = 0.0

        self._u_pid._integral = 0.0
        self._u_pid._prev_error = 0.0

        self.pre_dive_start_time = None
        self.pre_dive_complete = False
        self.gimbal_locked = False
        self.pitch_alignment_start = None
        self.pre_dive_initial_gimbal_pitch = 0.0
        self.gimbal_target_pitch = 0.0
        self._b2_first_satisfied_time = None
        self.pre_dive_triggered = False
        self._pre_dive_ramped_pitch_cmd = None
        self.prev_pre_dive_elevator = 0.0

        self._elevator_lpf_state = 0.0

        self._q_filt = 0.0
        self._r_filt = 0.0
        self._p_filt = 0.0
        self._prev_rudder_v14 = 0.0

        self.last_estimated_distance = None
        self.last_trigger_distance = None

        self.search_altitude_initialized = False
        self._frozen_trigger_slant = None
        self._frozen_trigger_ground = None
        self.current_mode = 'INIT'
        self.mode_history = []
        self.last_pitch_cmd = 0.0
        self.last_yaw_cmd = 0.0
        self.dynamic_alpha_est = self.alpha_bias
        self.dynamic_pitch_base = self.dive_angle - self.alpha_bias
        self.pre_dive_pitch_target = self.dynamic_pitch_base - self.pre_dive_overshoot

        print("[Controller] all controllers reset")

    def get_control_info(self):
        return {
            'mode': self.current_mode,
            'dive_angle': self.dive_angle,
            'pre_dive_target_pitch': self.pre_dive_target_pitch,
            'gimbal_target_pitch': self.gimbal_target_pitch,
            'gimbal_locked': self.gimbal_locked,
            'pre_dive_triggered': self.pre_dive_triggered,
            'last_estimated_distance': self.last_estimated_distance,
            'last_trigger_distance': self.last_trigger_distance,
        }


# ══════════════════════════════════════════════════════════════
#  Drift estimator
# ══════════════════════════════════════════════════════════════

class DriftEstimator:
    """drift = observed error rate - attitude-induced rate, EMA-smoothed."""

    def __init__(self, c_lon, c_lat, alpha=0.3):
        self.c_lon, self.c_lat, self.alpha = c_lon, c_lat, alpha
        self.drift_lon = 0.0
        self.drift_lat = 0.0
        self._prev_e_lon = None
        self._prev_e_lat = None
        self._prev_theta = None
        self._prev_psi = None

    def update(self, e_lon, e_lat, theta, psi, dt):
        if self._prev_e_lon is None:
            self._prev_e_lon, self._prev_e_lat = e_lon, e_lat
            self._prev_theta, self._prev_psi = theta, psi
            return
        if dt <= 0:
            return
        de_lon_obs = (e_lon - self._prev_e_lon) / dt
        de_lat_obs = (e_lat - self._prev_e_lat) / dt
        theta_dot = (theta - self._prev_theta) / dt
        psi_dot = (psi - self._prev_psi) / dt

        raw_lon = np.clip(de_lon_obs - self.c_lon * theta_dot, -0.5, 0.5)
        raw_lat = np.clip(de_lat_obs - (-self.c_lat * psi_dot), -0.5, 0.5)

        self.drift_lon = self.alpha * raw_lon + (1 - self.alpha) * self.drift_lon
        self.drift_lat = self.alpha * raw_lat + (1 - self.alpha) * self.drift_lat

        self._prev_e_lon, self._prev_e_lat = e_lon, e_lat
        self._prev_theta, self._prev_psi = theta, psi

    def get_drift(self):
        return self.drift_lon, self.drift_lat

    def reset(self):
        self.drift_lon = self.drift_lat = 0.0
        self._prev_e_lon = self._prev_e_lat = None
        self._prev_theta = self._prev_psi = None


# ══════════════════════════════════════════════════════════════
#  Target Kalman filter
# ══════════════════════════════════════════════════════════════

class TargetKalmanFilter:
    """CV-model KF, state [px, py, vx, vy] in NED."""

    def __init__(self, process_noise_accel=2.5, measurement_noise_pos=5.0,
                 init_pos_std=30.0, init_vel_std=15.0, max_speed_mps=40.0):
        self.q_accel = process_noise_accel
        self.r_pos = measurement_noise_pos
        self.init_pos_std = init_pos_std
        self.init_vel_std = init_vel_std
        self.max_speed_mps = max_speed_mps
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1e6
        self._last_update_time = None
        self._measurement_count = 0
        self._initialized = False
        self.buffer = []

    def _predict(self, dt):
        if dt <= 0:
            return
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        s2 = self.q_accel ** 2
        Q = s2 * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, ray_world, slant_range, ac_x_ned, ac_y_ned, timestamp):
        if ray_world is None or slant_range is None or slant_range <= 0:
            return
        if not np.isfinite(slant_range):
            return
        px = ac_x_ned + slant_range * ray_world[0]
        py = ac_y_ned + slant_range * ray_world[1]
        if not (np.isfinite(px) and np.isfinite(py)):
            return

        if not self._initialized:
            self.x[:2] = [px, py]
            self.x[2:] = [0, 0]
            self.P = np.diag([self.init_pos_std**2]*2 + [self.init_vel_std**2]*2)
            self._initialized = True
            self._last_update_time = timestamp
            self._measurement_count = 1
            self.buffer.append(timestamp)
            return

        dt = timestamp - self._last_update_time
        if dt > 0:
            self._predict(dt)
        self._last_update_time = timestamp

        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        R = np.eye(2) * self.r_pos**2
        z = np.array([px, py])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        # Mahalanobis gate
        if float(y.T @ S_inv @ y) > 16.0:
            return

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        speed = math.hypot(self.x[2], self.x[3])
        if speed > self.max_speed_mps:
            s = self.max_speed_mps / speed
            self.x[2] *= s
            self.x[3] *= s

        self._measurement_count += 1
        self.buffer.append(timestamp)
        if len(self.buffer) > 50:
            self.buffer.pop(0)

    def get_velocity(self):
        return float(self.x[2]), float(self.x[3])

    def get_velocity_std(self):
        return float(math.sqrt(max(0.0, self.P[2,2] + self.P[3,3])))

    def get_confidence(self, full_std=2.0, zero_std=8.0):
        if not self._initialized or self._measurement_count < 3:
            return 0.0
        std = self.get_velocity_std()
        if std <= full_std:
            return 1.0
        if std >= zero_std:
            return 0.0
        return 1.0 - (std - full_std) / (zero_std - full_std)

    @property
    def vx_ned(self):
        return float(self.x[2])
    @property
    def vy_ned(self):
        return float(self.x[3])
    @property
    def speed_mps(self):
        return math.hypot(self.x[2], self.x[3])
    @property
    def is_ready(self):
        return self._initialized and self._measurement_count >= 3

    def get_summary(self):
        std = self.get_velocity_std() if self._initialized else float('inf')
        return (f"v=({self.vx_ned:+.1f},{self.vy_ned:+.1f})m/s "
                f"|v|={self.speed_mps:.1f} std={std:.1f} n={self._measurement_count}")

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1e6
        self._last_update_time = None
        self._measurement_count = 0
        self._initialized = False
        self.buffer.clear()


# ══════════════════════════════════════════════════════════════
#  Coordinate helpers
# ══════════════════════════════════════════════════════════════
#
# Conventions:
#   vision  : pixel (u, v) in [0, W] x [0, H]
#   control : half-screen normalized offsets in [-1, 1]
#             lat_off = (u - cx) / (W/2),  lon_off = (v - cy) / (H/2)
#   MPC     : standard normalized (x_n, y_n)
#             x_n = (u - cx) / fx,  y_n = (v - cy) / fy

def half_screen_to_normalized(lateral_offset, longitudinal_offset,
                                screen_width, screen_height, fx, fy):
    """Half-screen normalized (-1..1) -> standard normalized (x_n, y_n)."""
    x_n = lateral_offset * (screen_width / 2.0) / fx
    y_n = longitudinal_offset * (screen_height / 2.0) / fy
    return x_n, y_n


# ══════════════════════════════════════════════════════════════
#  Depth smoothing
# ══════════════════════════════════════════════════════════════

class _DepthEMA:
    """EMA smoothing for slant range, holds last value on missing input."""
    def __init__(self, alpha=0.3, min_value=5.0):
        self.alpha = alpha
        self.min_value = min_value
        self.value = None

    def update(self, raw):
        if raw is None or raw < self.min_value:
            return self.value if self.value is not None else self.min_value
        if self.value is None:
            self.value = raw
        else:
            self.value = self.alpha * raw + (1 - self.alpha) * self.value
        return self.value

    def get(self):
        return self.value if self.value is not None else self.min_value

    def reset(self):
        self.value = None


# ══════════════════════════════════════════════════════════════
#  MPC parameters
# ══════════════════════════════════════════════════════════════

@dataclass
class DiveMPCParams:
    """MPC parameters: 3DOF point-mass + first-order command response."""

    # First-order command response time constants (engineering values, no sysid)
    tau_chi:   float = 1.5      # s, track angle (banked-turn accumulation is slow)
    tau_gamma: float = 1.0      # s, flight-path angle (direct elevator authority)

    # Energy dynamics: V_dot = -g sin(gamma) - k_drag V^2.
    # Pure kinematics under-predicts V by ~3 m/s over a 0.6 s horizon;
    # k_drag = 0.5 rho S C_D / m from aircraft spec (conservatively low).
    g_accel:  float = 9.81      # m/s^2
    k_drag:   float = 0.001     # /m, drag-to-mass ratio

    # KF reset-on-DIVE: ray projection accumulates large systematic bias
    # during PRE_DIVE; reset position AND velocity at DIVE entry with
    # inflated P so subsequent measurements re-converge quickly.
    kf_dive_reset_pos_std_m: float = 30.0
    kf_dive_reset_vel_std_mps: float = 15.0

    # Pitch ~ gamma + alpha_nom at the dive operating point (0 = no comp)
    alpha_nominal_deg: float = 0.0

    # Image cost (geometric projection)
    Z_min: float = 5.0                   # depth floor (guards 1/Z)
    Z_ema_alpha: float = 0.3

    # Cost weights. gamma anchor removed: pitch_cmd bounds do the limiting;
    # a large image_y weight at DIVE entry would flatten the dive.
    w_image_x: float = 800.0
    w_image_y: float = 400.0
    w_gamma_anchor: float = 0.0
    gamma_anchor_hold_time:  float = 0.0
    gamma_anchor_decay_time: float = 0.0
    w_gamma_anchor_min:      float = 0.0
    w_image_y_max:           float = 400.0

    # Command regularization (relative to mission target / initial chi)
    R_chi_cmd:   float = 5.0
    R_gamma_cmd: float = 0.5

    # Command smoothness (delta-u cost)
    R_du_chi:   float = 100.0
    R_du_gamma: float = 100.0

    terminal_weight_ratio: float = 5.0

    # Horizon / solver
    horizon: int = 6
    mpc_dt: float = 0.10
    n_substeps: int = 5
    dt_id: float = 0.02
    max_iter: int = 25
    mpc_solve_interval_ticks: int = 5

    # Output command bounds
    pitch_cmd_min_deg:        float = -60.0
    pitch_cmd_max_deg:        float = 20.0
    heading_cmd_max_dev_deg:  float = 8.0

    # Command slew per MPC step (servo-style rate limit; keeps loop stable
    # while letting a step command land within ~2 solve cycles)
    cmd_slew_pitch_deg:   float = 2.0
    cmd_slew_heading_deg: float = 1.0

    # Command first-order LPF (before slew): smooths solver jitter,
    # slew remains the hard last-resort limit
    cmd_lpf_alpha:        float = 0.55   # 0 = no filter, larger = smoother

    # Max chi deviation from initial within the horizon (rad bound)
    chi_dev_max_deg: float = 30.0

    # Terminal guard (image-based LoS below this AGL)
    terminal_guard_enabled: bool = False
    terminal_guard_agl_m:   float = 150.0

    # Mission
    dive_pitch_target_deg: float = -30.0

    # Drift / target KF
    drift_ema_alpha: float = 0.8
    lead_time: float = 0.3   # fixed velocity feed-forward lead (s)

    kf_process_noise_accel:    float = 2.5
    kf_measurement_noise_pos:  float = 5.0
    kf_init_pos_std:           float = 30.0
    kf_init_vel_std:           float = 15.0
    velocity_max_speed_mps:    float = 40.0
    kf_vel_std_full_confidence: float = 2.0
    kf_vel_std_zero_confidence: float = 8.0

    use_velocity_feedforward:    bool = True
    velocity_ff_priority_duration: float = 4.0
    warn_if_no_world_state_updates: bool = True

    # ── Terminal Visual Servo (TVS) ──
    # Additive pure-P correction on top of MPC output, driven directly by the
    # measured bbox center (bypasses target_kf). bbox area fraction s is a
    # dimensionless proximity proxy that schedules the weight beta; correction
    # limits scale with beta so authority grows only when close.
    tvs_enabled: bool = True

    # Proximity schedule: s = bbox_w*bbox_h / screen area
    tvs_s_enter: float = 0.0006   # beta leaves 0 (~AGL 75 m)
    tvs_s_full:  float = 0.0018   # beta reaches 1 (~AGL 39 m)
    tvs_s_ema_alpha: float = 0.3
    tvs_latch_once_entered: bool = True   # beta monotone non-decreasing

    # Proportional gains (normalized error -> deg; error saturates at +/-1)
    tvs_K_lat_deg: float = 6.0
    tvs_K_lon_deg: float = 12.0

    # Correction caps (scaled by beta)
    tvs_max_corr_heading_deg: float = 5.0
    tvs_max_corr_pitch_deg:   float = 20.0

    # Correction smoothing (independent of the MPC cache LPF/slew)
    tvs_corr_lpf_alpha: float = 0.2    # 1 = frozen, 0 = no filter
    tvs_corr_slew_deg:  float = 2.0

    # Dropout / edge protection
    tvs_hold_on_lost: bool = True       # hold last correction on YOLO dropout
    tvs_lost_hold_max_ticks: int = 50   # then decay to zero
    tvs_edge_margin_frac: float = 0.04  # bbox near screen edge -> down-weight
    tvs_edge_weight: float = 0.3

    # Relax heading deviation cap while TVS active in terminal phase
    tvs_terminal_heading_dev_deg: float = 12.0

    # ── Mid-phase error-driven beta floor ──
    # The longitudinal image error is a stable bias through the mid phase
    # (not just terminal); the s schedule alone starts too late. Within an
    # AGL window, a persistent smoothed error raises a floor on beta:
    # beta = max(s schedule, error floor). Terminal phase stays s-driven.
    tvs_err_drive_enabled: bool = True
    tvs_err_drive_agl_high: float = 90.0   # above: bbox too small, disabled
    tvs_err_drive_agl_low:  float = 35.0   # below: s schedule takes over
    tvs_err_drive_ema_alpha: float = 0.25
    tvs_err_drive_confirm_ticks: int = 3   # consecutive over-threshold ticks
    tvs_err_drive_deadband: float = 0.04   # |v_norm_ema| below -> no trigger
    tvs_err_drive_saturate: float = 0.15   # at/above -> floor = max_floor
    tvs_err_drive_max_floor: float = 0.55

    @property
    def gamma_target_rad(self):
        return math.radians(self.dive_pitch_target_deg - self.alpha_nominal_deg)

    @property
    def gamma_cmd_min_rad(self):
        return math.radians(self.pitch_cmd_min_deg - self.alpha_nominal_deg)

    @property
    def gamma_cmd_max_rad(self):
        return math.radians(self.pitch_cmd_max_deg - self.alpha_nominal_deg)


# ══════════════════════════════════════════════════════════════
#  CasADi rotation helpers
# ══════════════════════════════════════════════════════════════

def _ca_Ry(rad):
    """CasADi Ry (same convention as _Ry_np)."""
    c, s = ca.cos(rad), ca.sin(rad)
    return ca.vertcat(
        ca.horzcat(c,  0, -s),
        ca.horzcat(0,  1,  0),
        ca.horzcat(s,  0,  c),
    )


def _ca_Rz(rad):
    """CasADi Rz."""
    c, s = ca.cos(rad), ca.sin(rad)
    return ca.vertcat(
        ca.horzcat( c, s, 0),
        ca.horzcat(-s, c, 0),
        ca.horzcat( 0, 0, 1),
    )


# ══════════════════════════════════════════════════════════════
#  NLP solver
# ══════════════════════════════════════════════════════════════

class DiveMPCSolver:
    """
    Parameterized CasADi NLP: 3DOF point-mass + first-order command response.

    State (6):
        x = [Dx, Dy, h, V, chi_offset, gamma]
        Dx, Dy     : aircraft position relative to target, NED plane (m)
        h          : AGL (m; target on the ground at h=0)
        V          : speed (m/s)
        chi_offset : track-angle deviation from chi_initial (rad); starts at 0
                     each solve to avoid wrap-around issues
        gamma      : flight-path angle (rad, absolute)

    Control (2):
        u = [chi_cmd_offset, gamma_cmd] (rad); converted by the controller to
        (pitch_cmd, heading_cmd) for the inner loop.

    Parameter vector P (22):
        [x0(6), R_b2c_flat(9), chi_initial(1), u_prev(2),
         target_vx(1), target_vy(1), w_gamma_anchor_eff(1), w_image_y_eff(1)]
        Caller supplies x0 with chi_offset=0 and u_prev already offset.
    """

    NX = 6
    NU = 2

    def __init__(self, params: DiveMPCParams):
        self.params = params
        self.N = params.horizon
        self.n_sub = params.n_substeps
        self.dt_sub = params.dt_id

        if not math.isclose(params.mpc_dt, self.n_sub * self.dt_sub, rel_tol=0.01):
            print(f"[MPC] warning: mpc_dt={params.mpc_dt} != n_sub*dt_id="
                  f"{self.n_sub * self.dt_sub}")

        self._warm_x0 = None
        self._warm_lam_g0 = None
        self._warm_lam_x0 = None

        self._build_solver()

        print(f"\n[MPC] solver built")
        print(f"  Horizon: {self.N} x {params.mpc_dt}s = {self.N * params.mpc_dt:.1f}s")
        print(f"  Sub-steps: {self.n_sub} x {self.dt_sub}s")
        print(f"  State dim: {self.NX} [Dx, Dy, h, V, chi_off, gamma]")
        print(f"  Input dim: {self.NU} [chi_cmd_off, gamma_cmd]")
        print(f"  tau_chi = {params.tau_chi:.2f}s, tau_gamma = {params.tau_gamma:.2f}s")
        print(f"  Energy: V_dot = -g sin(gamma) - k_D V^2, k_D={params.k_drag:.4f}/m")
        print(f"  Weights: w_image_x={params.w_image_x}, w_image_y={params.w_image_y}, "
              f"w_gamma_anchor={params.w_gamma_anchor}")
        print(f"  gamma bound: [{params.gamma_cmd_min_rad*180/math.pi:.0f}, "
              f"{params.gamma_cmd_max_rad*180/math.pi:.0f}] deg")
        print(f"  gamma_target = {math.degrees(params.gamma_target_rad):+.1f} deg "
              f"(pitch_cmd = {params.dive_pitch_target_deg:+.1f} deg)")
        print(f"  alpha_nominal = {params.alpha_nominal_deg:+.1f} deg")

    # ────────────────────────────────────────────────────
    #  Dynamics (CasADi symbolic)
    # ────────────────────────────────────────────────────

    def _step_dynamics(self, x_state, u, chi_initial, target_vx, target_vy):
        """
        One Euler sub-step. Kinematics + first-order command lag + energy
        equation for V (V_dot = -g sin(gamma) - k_D V^2).
        """
        p = self.params
        dt = self.dt_sub

        Dx          = x_state[0]
        Dy          = x_state[1]
        h           = x_state[2]
        V           = x_state[3]
        chi_offset  = x_state[4]
        gamma       = x_state[5]

        chi_cmd_offset = u[0]
        gamma_cmd      = u[1]

        # First-order command response
        chi_offset_next = chi_offset + dt * (chi_cmd_offset - chi_offset) / p.tau_chi
        gamma_next      = gamma      + dt * (gamma_cmd      - gamma)      / p.tau_gamma

        chi_actual_next = chi_initial + chi_offset_next

        cg = ca.cos(gamma_next)
        sg = ca.sin(gamma_next)
        cc = ca.cos(chi_actual_next)
        sc = ca.sin(chi_actual_next)

        # Energy dynamics
        V_dot = -p.g_accel * sg - p.k_drag * V * V
        V_next = V + dt * V_dot
        V_next = ca.fmax(V_next, 5.0)   # soft numeric floor

        # Delta = aircraft - target, so d(Delta)/dt = ac_vel - target_vel
        ac_vx = V_next * cg * cc
        ac_vy = V_next * cg * sc
        ac_vh = V_next * sg

        Dx_next = Dx + dt * (ac_vx - target_vx)
        Dy_next = Dy + dt * (ac_vy - target_vy)
        h_next  = h  + dt * ac_vh

        return ca.vertcat(Dx_next, Dy_next, h_next, V_next, chi_offset_next, gamma_next)

    def _multi_step(self, x_state, u, chi_initial, target_vx, target_vy):
        """One MPC step (mpc_dt) = n_sub sub-steps (ZOH on u)."""
        x = x_state
        for _ in range(self.n_sub):
            x = self._step_dynamics(x, u, chi_initial, target_vx, target_vy)
        return x

    # ────────────────────────────────────────────────────
    #  Image projection (geometric, for the cost)
    # ────────────────────────────────────────────────────

    def _project_to_image(self, x_state, R_b2c, chi_initial):
        """Project (Dx, Dy, h, chi_offset, gamma) to image plane (x_n, y_n)."""
        p = self.params

        Dx          = x_state[0]
        Dy          = x_state[1]
        h           = x_state[2]
        chi_offset  = x_state[4]
        gamma       = x_state[5]

        chi_actual = chi_initial + chi_offset

        v_ned = ca.vertcat(-Dx, -Dy, h)

        alpha_nom_rad = math.radians(p.alpha_nominal_deg)
        R_ned2body = ca.mtimes(_ca_Ry(gamma + alpha_nom_rad), _ca_Rz(chi_actual))
        v_body = ca.mtimes(R_ned2body, v_ned)

        v_cam = ca.mtimes(R_b2c, v_body)

        Vcx, Vcy, Vcz = v_cam[0], v_cam[1], v_cam[2]

        Vcz_safe = ca.fmax(Vcz, p.Z_min)
        x_n = Vcx / Vcz_safe
        y_n = Vcy / Vcz_safe

        return x_n, y_n

    # ────────────────────────────────────────────────────
    #  NLP construction
    # ────────────────────────────────────────────────────

    def _build_solver(self):
        p = self.params
        N = self.N

        # Note: ca.reshape is column-major, numpy flatten() row-major -> .T
        P = ca.MX.sym('P', 22)
        x0_p     = P[0:6]
        R_b2c_p  = ca.reshape(P[6:15], 3, 3).T
        chi_init = P[15]
        u_prev_p = P[16:18]
        target_vx_p = P[18]
        target_vy_p = P[19]
        w_gamma_anchor_eff = P[20]
        w_image_y_eff      = P[21]

        gamma_target = p.gamma_target_rad
        chi_dev_max_rad = math.radians(p.chi_dev_max_deg)

        w, w0, lbw, ubw, g, lbg, ubg = [], [], [], [], [], [], []
        J = 0
        Xk = x0_p
        Uk_prev = u_prev_p

        for k in range(N):
            # Control U_k = [chi_cmd_offset, gamma_cmd] (rad)
            Uk = ca.MX.sym(f'U_{k}', self.NU)
            w.append(Uk)
            lbw += [-chi_dev_max_rad, p.gamma_cmd_min_rad]
            ubw += [ chi_dev_max_rad, p.gamma_cmd_max_rad]
            w0  += [0.0, gamma_target]

            # Command regularization (toward chi_offset=0 and gamma_target)
            J += p.R_chi_cmd   * Uk[0]**2
            J += p.R_gamma_cmd * (Uk[1] - gamma_target)**2

            # Delta-u cost
            dU = Uk - Uk_prev
            J += p.R_du_chi   * dU[0]**2
            J += p.R_du_gamma * dU[1]**2

            # Predict next state
            Xk_next = self._multi_step(Xk, Uk, chi_init, target_vx_p, target_vy_p)

            # New state variable X_{k+1}
            Xk1 = ca.MX.sym(f'X_{k+1}', self.NX)
            w.append(Xk1)
            gamma_state_min = p.gamma_cmd_min_rad - math.radians(20.0)
            gamma_state_max = p.gamma_cmd_max_rad + math.radians(20.0)
            lbw += [-5000.0, -5000.0, 0.0,  10.0, -chi_dev_max_rad, gamma_state_min]
            ubw += [ 5000.0,  5000.0, 1000.0, 100.0, chi_dev_max_rad, gamma_state_max]
            w0  += [0.0, 0.0, 200.0, 50.0, 0.0, gamma_target]

            # Dynamics equality constraint
            g.append(Xk1 - Xk_next)
            lbg += [0.0] * self.NX
            ubg += [0.0] * self.NX

            # Stage cost
            x_n_k1, y_n_k1 = self._project_to_image(Xk1, R_b2c_p, chi_init)

            J += p.w_image_x * x_n_k1**2 + w_image_y_eff * y_n_k1**2

            gamma_k1 = Xk1[5]
            J += w_gamma_anchor_eff * (gamma_k1 - gamma_target)**2

            Xk = Xk1
            Uk_prev = Uk

        # Terminal cost
        x_n_N, y_n_N = self._project_to_image(Xk, R_b2c_p, chi_init)
        J += (p.terminal_weight_ratio - 1.0) * (
              p.w_image_x * x_n_N**2 + w_image_y_eff * y_n_N**2)

        # NLP
        w_cat = ca.vertcat(*w)
        g_cat = ca.vertcat(*g)
        opts = {
            'ipopt.print_level': 0,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.max_iter': p.max_iter,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 3,
            'print_time': 0,
            'ipopt.sb': 'yes',
        }
        self.solver = ca.nlpsol('dive_mpc', 'ipopt',
                                {'f': J, 'x': w_cat, 'g': g_cat, 'p': P}, opts)

        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        self._w0_default = w0
        self._stride = self.NU + self.NX

        print(f"  NLP: {w_cat.shape[0]} vars, {g_cat.shape[0]} equality constraints")

    # ────────────────────────────────────────────────────
    #  Solve API
    # ────────────────────────────────────────────────────

    def solve(self, x0_full, R_b2c, u_prev_full, chi_initial,
              target_vx=0.0, target_vy=0.0,
              w_gamma_anchor_eff=None, w_image_y_eff=None):
        """
        Solve one MPC step.

        Args:
            x0_full: (6,) [Dx, Dy, h, V, chi_actual_rad, gamma_rad];
                     absolute chi, converted internally to offset.
            u_prev_full: (2,) previous [chi_cmd_actual_rad, gamma_cmd_rad].
            chi_initial: current chi (rad), the global reference.
            target_vx/vy: target NED velocity (m/s); the NLP animates the target.
            w_gamma_anchor_eff / w_image_y_eff: scheduled weights (None = defaults).

        Returns:
            (chi_cmd_rad_actual, gamma_cmd_rad, info)
        """
        t0 = time.time()
        p = self.params

        x0_arr = np.asarray(x0_full, dtype=float).flatten()
        assert x0_arr.shape == (6,), f"x0 shape: {x0_arr.shape}"

        u_prev_arr = np.asarray(u_prev_full, dtype=float).flatten()
        assert u_prev_arr.shape == (2,), f"u_prev shape: {u_prev_arr.shape}"

        chi_initial = float(chi_initial)

        # Absolute chi -> offset
        x0_offset = x0_arr.copy()
        x0_offset[4] = 0.0    # chi_offset starts at 0

        u_prev_offset = u_prev_arr.copy()
        chi_cmd_prev_offset = u_prev_arr[0] - chi_initial
        # Wrap to [-pi, pi] to avoid cold-start wrap issues
        while chi_cmd_prev_offset > math.pi:
            chi_cmd_prev_offset -= 2 * math.pi
        while chi_cmd_prev_offset < -math.pi:
            chi_cmd_prev_offset += 2 * math.pi
        chi_dev_max_rad = math.radians(p.chi_dev_max_deg)
        chi_cmd_prev_offset = float(np.clip(chi_cmd_prev_offset,
                                             -chi_dev_max_rad, chi_dev_max_rad))
        u_prev_offset[0] = chi_cmd_prev_offset

        # Clip target velocity (guard against KF spikes)
        target_vx = float(np.clip(target_vx, -50.0, 50.0))
        target_vy = float(np.clip(target_vy, -50.0, 50.0))

        _w_ga = w_gamma_anchor_eff if w_gamma_anchor_eff is not None else p.w_gamma_anchor
        _w_iy = w_image_y_eff if w_image_y_eff is not None else p.w_image_y

        P_val = np.concatenate([
            x0_offset,                                  # 6
            np.asarray(R_b2c, dtype=float).flatten(),  # 9
            [chi_initial],                              # 1
            u_prev_offset,                              # 2
            [target_vx, target_vy],                     # 2
            [_w_ga, _w_iy],                             # 2
        ])
        assert P_val.shape == (22,), f"P_val shape: {P_val.shape}"

        # Warm start
        if self._warm_x0 is not None:
            w0_init = self._warm_x0
        else:
            w0_init = []
            for k in range(self.N):
                w0_init += [0.0, p.gamma_target_rad]   # U_k
                w0_init += [
                    float(x0_offset[0]), float(x0_offset[1]), float(x0_offset[2]),
                    float(x0_offset[3]), 0.0, p.gamma_target_rad,
                ]   # X_{k+1}

        kwargs = {
            'x0': w0_init,
            'lbx': self.lbw,
            'ubx': self.ubw,
            'lbg': self.lbg,
            'ubg': self.ubg,
            'p': P_val,
        }
        if self._warm_lam_g0 is not None:
            kwargs['lam_g0'] = self._warm_lam_g0
        if self._warm_lam_x0 is not None:
            kwargs['lam_x0'] = self._warm_lam_x0

        sol = self.solver(**kwargs)
        w_opt = sol['x'].full().flatten()

        # Step-0 command (offset -> absolute chi)
        u_first_offset = w_opt[0:self.NU]
        chi_cmd_offset = float(u_first_offset[0])
        gamma_cmd_rad  = float(u_first_offset[1])
        chi_cmd_rad    = chi_initial + chi_cmd_offset

        # Predicted step-1 state
        x1 = w_opt[self.NU : self.NU + self.NX]
        Dx_pred = float(x1[0])
        Dy_pred = float(x1[1])
        h_pred = float(x1[2])
        chi_offset_pred = float(x1[4])
        gamma_pred = float(x1[5])

        # Warm-start shift
        stride = self._stride
        shifted = np.zeros_like(w_opt)
        for k in range(self.N - 1):
            shifted[k * stride:(k + 1) * stride] = w_opt[(k + 1) * stride:(k + 2) * stride]
        shifted[(self.N - 1) * stride:] = w_opt[(self.N - 1) * stride:]
        self._warm_x0 = shifted.tolist()
        try:
            self._warm_lam_g0 = sol['lam_g'].full().flatten().tolist()
            self._warm_lam_x0 = sol['lam_x'].full().flatten().tolist()
        except Exception:
            pass

        info = {
            'solve_time': time.time() - t0,
            'cost': float(sol['f']),
            'status': str(self.solver.stats()['return_status']),
            'u_first_offset':   u_first_offset.copy(),
            'u_first_absolute': np.array([chi_cmd_rad, gamma_cmd_rad]),
            'chi_cmd_offset_deg': math.degrees(chi_cmd_offset),
            'chi_cmd_deg':        math.degrees(chi_cmd_rad),
            'gamma_cmd_deg':      math.degrees(gamma_cmd_rad),
            'Dx_pred': Dx_pred,
            'Dy_pred': Dy_pred,
            'h_pred':  h_pred,
            'chi_offset_pred_deg': math.degrees(chi_offset_pred),
            'gamma_pred_deg':      math.degrees(gamma_pred),
        }
        return chi_cmd_rad, gamma_cmd_rad, info

    def reset(self):
        self._warm_x0 = None
        self._warm_lam_g0 = None
        self._warm_lam_x0 = None


# ══════════════════════════════════════════════════════════════
#  MPC dive controller
# ══════════════════════════════════════════════════════════════

class MPCDiveController(GimbalIntegratedDiveController):
    """
    MPC dive controller: 3DOF point-mass MPC + cascade inner loop.

    API:
      - control(state, ...)
      - update_target_world_state(...)
      - update_measured_image(...) / get_tvs_debug()
      - notify_dive_started(...) / notify_gimbal_locked(...)
      - update_dkf_drift(...)
      - reset_controllers()
    """

    def __init__(self, *args,
                 mpc_params=None,
                 ray_estimator=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if mpc_params is None:
            mpc_params = DiveMPCParams()
        self.mpc_params = mpc_params

        if ray_estimator is None:
            raise ValueError("[MPCDiveController] ray_estimator is required")
        self.ray_estimator = ray_estimator

        # Core components
        self.mpc_solver = DiveMPCSolver(mpc_params)
        self.depth_ema = _DepthEMA(
            alpha=mpc_params.Z_ema_alpha, min_value=mpc_params.Z_min)

        self.drift_estimator = DriftEstimator(
            c_lon=0.025, c_lat=0.033,
            alpha=mpc_params.drift_ema_alpha)
        self.target_kf = TargetKalmanFilter(
            process_noise_accel=mpc_params.kf_process_noise_accel,
            measurement_noise_pos=mpc_params.kf_measurement_noise_pos,
            init_pos_std=mpc_params.kf_init_pos_std,
            init_vel_std=mpc_params.kf_init_vel_std,
            max_speed_mps=mpc_params.velocity_max_speed_mps)

        # Aircraft / environment cache
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
        # ray_world for the KF-independent terminal guard LoS
        self._last_ray_world = None
        self._last_ray_timestamp = None

        # Locked at DIVE entry
        self._dive_start_time = None
        self.gimbal_pitch_locked = 0.0
        self.gimbal_yaw_locked = 0.0
        self.R_b2c_locked = compute_R_b2c(0.0, 0.0)

        # MPC outer-loop state: u_prev is (chi_cmd, gamma_cmd) in rad
        self._mpc_u_prev = np.zeros(2)
        self._mpc_solve_times = deque(maxlen=50)
        self._mpc_solve_count = 0
        self._mpc_last_info = None

        # Low-rate MPC throttle + cached command
        self._mpc_step_counter = mpc_params.mpc_solve_interval_ticks
        self._mpc_cached_pitch_cmd = None
        self._mpc_cached_heading_offset_deg = 0.0
        self._mpc_cached_heading_cmd = None
        self._last_pitch_cmd_for_log = self.dive_angle
        self._last_yaw_cmd_for_log = 0.0
        self._world_state_update_count = 0

        # DKF drift (legacy, unused by MPC)
        self._dkf_drift_lon = 0.0
        self._dkf_drift_lat = 0.0
        self._dkf_drift_valid = False

        self._terminal_guard_announced = False

        # KF reset-on-DIVE
        self._kf_rejected_count = 0
        self._kf_dive_reset_pending = False
        self._kf_dive_reset_done_count = 0

        # Shared DIVE inner loop: tune inner_loop_controller.py.
        init_dive_inner_loop(self)

        # Manual inner-loop test mode: skip MPC + vision + mode machine,
        # hold altitude with search_control, then fixed pitch_cmd into the
        # inner loop. Usage:
        #   controller.manual_mode_enabled = True
        #   controller.manual_dive_pitch_deg = -30.0
        #   controller.manual_trigger_alt_agl = None  # None = dive immediately
        self.manual_mode_enabled = False
        self.manual_dive_pitch_deg = -30.0
        self.manual_trigger_alt_agl = None
        self._manual_initial_heading = None
        self._manual_initial_alt = None
        self._manual_dive_armed = False

        # Terminal Visual Servo (TVS) state; updated by update_measured_image(),
        # consumed by _apply_terminal_visual_servo() in dive_control paths.
        self._tvs_u_norm = 0.0          # bbox lateral normalized error [-1,1]
        self._tvs_v_norm = 0.0          # bbox longitudinal normalized error
        self._tvs_s_raw = 0.0           # bbox screen-area fraction (raw)
        self._tvs_s_ema = 0.0
        self._tvs_beta = 0.0            # proximity-scheduled weight [0,1]
        self._tvs_entered = False       # latch flag
        self._tvs_detected = False
        self._tvs_edge_flagged = False
        self._tvs_lost_ticks = 0
        self._tvs_corr_pitch_deg = 0.0    # applied correction (LPF/slew state)
        self._tvs_corr_heading_deg = 0.0
        self._tvs_active = False
        self._tvs_last_update_count = 0
        # Mid-phase error-driven beta floor state
        self._tvs_v_norm_ema = 0.0
        self._tvs_err_confirm_count = 0
        self._tvs_err_floor = 0.0       # diagnostic
        self._tvs_beta_sched = 0.0      # pure s-schedule beta (diagnostic)

        print(f"\n[MPCDiveController] loaded")
        print(f"  arch: 3DOF point-mass + first-order command response + cascade inner loop")
        print(f"  Camera: fx={ray_estimator.fx:.1f} fy={ray_estimator.fy:.1f}, "
              f"screen={ray_estimator.screen_width}x{ray_estimator.screen_height}")
        print(f"  MPC weights: w_image_x={mpc_params.w_image_x} "
              f"w_image_y={mpc_params.w_image_y} w_gamma_anchor={mpc_params.w_gamma_anchor}")
        print(f"  pitch_cmd range: [{mpc_params.pitch_cmd_min_deg}, {mpc_params.pitch_cmd_max_deg}] deg")
        print(f"  lateral: hdg_slew={mpc_params.cmd_slew_heading_deg}/step "
              f"hdg_max_dev={mpc_params.heading_cmd_max_dev_deg}")
        print(f"  MPC update rate: every {mpc_params.mpc_solve_interval_ticks} ticks "
              f"(~{1.0/(mpc_params.mpc_solve_interval_ticks * 0.01):.0f} Hz)")
        print(f"  Cmd slew: pitch={mpc_params.cmd_slew_pitch_deg}/step, "
              f"heading={mpc_params.cmd_slew_heading_deg}/step")
        tau_ms = 0.0 if mpc_params.cmd_lpf_alpha <= 0 else \
                 -mpc_params.mpc_solve_interval_ticks * 10.0 / math.log(mpc_params.cmd_lpf_alpha)
        print(f"  cmd LPF: alpha={mpc_params.cmd_lpf_alpha} (tau~{tau_ms:.0f}ms)")
        print(f"  KF reset-on-DIVE: P_std={mpc_params.kf_dive_reset_pos_std_m}m")
        print(f"  rudder slew: {self.rudder_slew_v22}/tick, "
              f"LPF alpha={self.rudder_lpf_alpha} "
              f"({'OFF' if self.rudder_lpf_alpha >= 0.99 else 'ON'})")
        print(f"  aileron LPF alpha={self.aileron_lpf_alpha} "
              f"({'OFF' if self.aileron_lpf_alpha >= 0.99 else 'ON'})")
        print(f"  TVS (terminal visual servo): {'ON' if mpc_params.tvs_enabled else 'OFF'}")
        if mpc_params.tvs_enabled:
            print(f"    schedule: s_enter={mpc_params.tvs_s_enter:.4f} "
                  f"s_full={mpc_params.tvs_s_full:.4f} (EMA alpha={mpc_params.tvs_s_ema_alpha})")
            print(f"    gains: K_lat={mpc_params.tvs_K_lat_deg}/unit "
                  f"K_lon={mpc_params.tvs_K_lon_deg}/unit")
            print(f"    corr caps (x beta): heading={mpc_params.tvs_max_corr_heading_deg} "
                  f"pitch={mpc_params.tvs_max_corr_pitch_deg}")
            print(f"    corr LPF alpha={mpc_params.tvs_corr_lpf_alpha} "
                  f"slew={mpc_params.tvs_corr_slew_deg}/tick")
            print(f"    hold on lost={mpc_params.tvs_hold_on_lost} "
                  f"(max {mpc_params.tvs_lost_hold_max_ticks} ticks), "
                  f"terminal hdg dev -> {mpc_params.tvs_terminal_heading_dev_deg}")
            print(f"    error-driven beta floor: "
                  f"{'ON' if mpc_params.tvs_err_drive_enabled else 'OFF'}")
            if mpc_params.tvs_err_drive_enabled:
                print(f"      AGL window: {mpc_params.tvs_err_drive_agl_low}~"
                      f"{mpc_params.tvs_err_drive_agl_high}m")
                print(f"      error signal: v_norm EMA alpha={mpc_params.tvs_err_drive_ema_alpha}, "
                      f"{mpc_params.tvs_err_drive_confirm_ticks}-tick confirm")
                print(f"      mapping: |v_norm_ema| {mpc_params.tvs_err_drive_deadband}"
                      f"->{mpc_params.tvs_err_drive_saturate} linear to floor 0"
                      f"->{mpc_params.tvs_err_drive_max_floor}")

    # ════════════════════════════════════════════
    #  Public API
    # ════════════════════════════════════════════

    def control(self, state, target_detected, lateral_offset, longitudinal_offset,
                x_position, current_time,
                gimbal_pitch=0.0, gimbal_yaw=0.0, target_centered=False,
                estimated_distance=None, distance_confidence=0.0,
                y_position=0.0):
        """Override: cache state/gimbal, then dispatch to the base state machine."""
        self._latest_state = state
        self._latest_gimbal_pitch = gimbal_pitch
        self._latest_gimbal_yaw = gimbal_yaw

        # Manual mode short-circuit: skip MPC + vision + mode machine
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

    def _manual_test_control(self, state, current_time):
        """
        Manual mode: SEARCH holds altitude via search_control; DIVE feeds a
        fixed pitch_cmd + locked heading straight into the inner loop.
        Trigger: AGL < manual_trigger_alt_agl, or None = immediately.

        Returns: (elevator, aileron, rudder, throttle, pitch_cmd, yaw_cmd)
        """
        airspeed     = state[0]
        p_rate       = state[7]
        q_rate       = state[8]
        r_rate       = state[9]
        roll         = state[10]
        pitch        = state[11]
        heading      = state[12]
        altitude_msl = state[19]
        altitude_agl = altitude_msl - self._runway_elev

        # First call: lock initial conditions
        if self._manual_initial_heading is None:
            self._manual_initial_heading = heading
            self._manual_initial_alt = altitude_agl
            print(f"\n{'='*60}")
            print(f"[MANUAL TEST] started (MPC + vision bypassed)")
            print(f"  initial heading = {heading:+.1f} (locked)")
            print(f"  initial AGL     = {altitude_agl:.1f} m")
            print(f"  dive pitch cmd  = {self.manual_dive_pitch_deg:+.1f}")
            if self.manual_trigger_alt_agl is None:
                print(f"  trigger: immediate DIVE")
                self._manual_dive_armed = True
            else:
                print(f"  trigger: AGL < {self.manual_trigger_alt_agl} m")
            print(f"{'='*60}\n")

        # Trigger check
        if not self._manual_dive_armed:
            if (self.manual_trigger_alt_agl is not None and
                    altitude_agl < self.manual_trigger_alt_agl):
                self._manual_dive_armed = True
                print(f"\n[MANUAL TEST] DIVE @ AGL={altitude_agl:.1f}m, "
                      f"pitch_cmd -> {self.manual_dive_pitch_deg:+.1f}\n")

        # Control
        if not self._manual_dive_armed:
            rudder, elevator, aileron, pitch_cmd, yaw_cmd = self.search_control(
                pitch, roll, heading, altitude_agl,
                gimbal_yaw_cmd=0.0,
                current_p=p_rate, current_r=r_rate)
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

    def update_target_world_state(self, ray_world, slant_range, ground_range,
                                    ac_x_ned, ac_y_ned, ac_vx_ned, ac_vy_ned,
                                    altitude_agl, aircraft_heading_rad, timestamp):
        """
        Feed the target KF and cache aircraft NED position/velocity. Call every
        tick.

        KF reset-on-DIVE: ray projection accumulates a large systematic bias
        during PRE_DIVE and the KF can lock onto a wrong position (its
        Mahalanobis gate then rejects the correct measurements). At DIVE entry
        the first measurement overwrites position AND velocity with inflated P;
        after that the KF is trusted fully.
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
        self._last_ray_world = (float(ray_world[0]), float(ray_world[1]), float(ray_world[2]))
        self._last_ray_timestamp = timestamp
        self._world_state_update_count += 1
        self.depth_ema.update(slant_range)

        # DIVE-entry KF reset (position + velocity)
        if self._kf_dive_reset_pending and self.target_kf._initialized:
            meas_target_x = ac_x_ned + slant_range * float(ray_world[0])
            meas_target_y = ac_y_ned + slant_range * float(ray_world[1])
            if np.isfinite(meas_target_x) and np.isfinite(meas_target_y):
                old_x = float(self.target_kf.x[0])
                old_y = float(self.target_kf.x[1])
                old_vx = float(self.target_kf.x[2])
                old_vy = float(self.target_kf.x[3])
                self.target_kf.x[0] = meas_target_x
                self.target_kf.x[1] = meas_target_y
                # Velocity is the derivative of a wrong position: reset it.
                # The KF re-accumulates true velocity within ~1 s of DIVE.
                self.target_kf.x[2] = 0.0
                self.target_kf.x[3] = 0.0
                p = self.mpc_params
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
                shift = math.hypot(meas_target_x - old_x, meas_target_y - old_y)
                old_speed = math.hypot(old_vx, old_vy)
                print(f"  [KF reset] DIVE entry: target ({old_x:.0f},{old_y:.0f}) -> "
                      f"({meas_target_x:.0f},{meas_target_y:.0f}), "
                      f"shift={shift:.0f}m, velocity reset (old |v|={old_speed:.1f} m/s)")
                # This measurement served as the cold-init; skip update()
                return

        self.target_kf.update(ray_world, slant_range, ac_x_ned, ac_y_ned, timestamp)

    def notify_dive_started(self, timestamp=None):
        """PRE_DIVE -> DIVE transition."""
        if timestamp is None:
            timestamp = time.time()
        self._dive_start_time = timestamp

        # Seed the command cache with the current pitch: PRE_DIVE overshoots
        # below dive_target, so starting the cache at dive_target would give
        # the inner loop a pull-up error at the transition.
        if self._latest_state is not None:
            current_pitch = float(self._latest_state[11])
            self._mpc_cached_pitch_cmd = current_pitch
            _cache_init_msg = (f"current pitch {current_pitch:+.1f} "
                               f"(dive_target={self.mpc_params.dive_pitch_target_deg:+.1f}, "
                               f"diff {current_pitch - self.mpc_params.dive_pitch_target_deg:+.1f})")
        else:
            self._mpc_cached_pitch_cmd = self.mpc_params.dive_pitch_target_deg
            _cache_init_msg = f"dive_target {self.mpc_params.dive_pitch_target_deg:+.1f} (fallback, no state)"
        self._mpc_cached_heading_offset_deg = 0.0
        self._mpc_cached_heading_cmd = None   # absolute heading target (locked at solve)
        self._mpc_step_counter = self.mpc_params.mpc_solve_interval_ticks
        # u_prev init: chi_cmd = current chi, gamma_cmd = gamma_target
        self._mpc_u_prev = np.array([
            self._last_ac_heading_rad,
            self.mpc_params.gamma_target_rad,
        ])
        self._terminal_guard_announced = False
        self._kf_rejected_count = 0
        # Next measurement resets the KF
        self._kf_dive_reset_pending = True
        self.mpc_solver.reset()

        print(f"\n[MPC] DIVE entered. First solve on next DIVE tick "
              f"(every {self.mpc_params.mpc_solve_interval_ticks} ticks)")
        print(f"      initial pitch_cmd = {_cache_init_msg}")

        if (self.mpc_params.warn_if_no_world_state_updates
                and self._world_state_update_count < 5):
            print(f"[MPC] warning: update_target_world_state() called only "
                  f"{self._world_state_update_count} times")

    def notify_gimbal_locked(self, timestamp=None, gimbal_pitch=0.0, gimbal_yaw=None):
        """Lock R_b2c at gimbal lock (angles from caller; default 0/0)."""
        if timestamp is None:
            timestamp = time.time()

        if gimbal_yaw is None:
            gimbal_yaw = 0.0

        self.gimbal_pitch_locked = float(gimbal_pitch)
        self.gimbal_yaw_locked = float(gimbal_yaw)
        self.R_b2c_locked = compute_R_b2c(self.gimbal_pitch_locked, self.gimbal_yaw_locked)

        elapsed = timestamp - self._dive_start_time if self._dive_start_time else 0
        print(f"[MPC] gimbal locked at "
              f"({self.gimbal_pitch_locked:+.1f}, {self.gimbal_yaw_locked:+.1f}) "
              f"(DIVE +{elapsed:.1f}s)")

    def update_dkf_drift(self, u_dot_px, v_dot_px,
                          pitch_rate_deg, heading_rate_deg,
                          screen_width=1920, screen_height=1010):
        """DKF drift is not used by the MPC; signature kept for the main loop."""
        self._dkf_drift_lon = v_dot_px / max(screen_height / 2.0, 1.0)
        self._dkf_drift_lat = u_dot_px / max(screen_width / 2.0, 1.0)
        self._dkf_drift_valid = True

    # ════════════════════════════════════════════
    #  Terminal Visual Servo (TVS)
    # ════════════════════════════════════════════

    def update_measured_image(self, cx, cy, bbox_w, bbox_h,
                               screen_w, screen_h, detected, timestamp=None,
                               agl=None):
        """
        Update TVS state from the raw bbox every tick (measured image error +
        proximity schedule). Bypasses target_kf and depth estimation entirely;
        the bbox area fraction is only a dimensionless proximity proxy.

        Args:
            agl: current AGL (m) for the error-driven-floor window;
                 None disables the error drive.
        """
        p = self.mpc_params
        self._tvs_last_update_count += 1

        # Dropout handling
        if not detected:
            self._tvs_detected = False
            if p.tvs_hold_on_lost:
                self._tvs_lost_ticks += 1
                if self._tvs_lost_ticks > p.tvs_lost_hold_max_ticks:
                    # Held too long -> decay beta to zero
                    decay = max(0.0, 1.0 - (self._tvs_lost_ticks
                                            - p.tvs_lost_hold_max_ticks) / 25.0)
                    self._tvs_beta *= decay
                # else: hold last u_norm / v_norm / beta
            else:
                self._tvs_beta = 0.0
            return

        self._tvs_detected = True
        self._tvs_lost_ticks = 0

        half_w = max(screen_w / 2.0, 1.0)
        half_h = max(screen_h / 2.0, 1.0)

        # Normalized errors in [-1, 1]
        u_norm = float(np.clip((cx - half_w) / half_w, -1.0, 1.0))
        v_norm = float(np.clip((cy - half_h) / half_h, -1.0, 1.0))
        self._tvs_u_norm = u_norm
        self._tvs_v_norm = v_norm

        # bbox screen-area fraction (proximity proxy)
        screen_area = max(screen_w * screen_h, 1.0)
        s_raw = float(max(bbox_w, 0.0) * max(bbox_h, 0.0)) / screen_area
        self._tvs_s_raw = s_raw

        a = p.tvs_s_ema_alpha
        if self._tvs_s_ema <= 0.0:
            self._tvs_s_ema = s_raw       # cold start
        else:
            self._tvs_s_ema = a * s_raw + (1.0 - a) * self._tvs_s_ema

        # beta from the s schedule
        denom = max(p.tvs_s_full - p.tvs_s_enter, 1e-6)
        beta_sched = float(np.clip(
            (self._tvs_s_ema - p.tvs_s_enter) / denom, 0.0, 1.0))
        self._tvs_beta_sched = beta_sched

        # Mid-phase error-driven beta floor.
        # v_norm EMA is updated every tick (not only inside the window).
        ea = p.tvs_err_drive_ema_alpha
        if self._tvs_last_update_count <= 1:
            self._tvs_v_norm_ema = v_norm   # cold start
        else:
            self._tvs_v_norm_ema = ea * v_norm + (1.0 - ea) * self._tvs_v_norm_ema

        err_floor = 0.0
        if (p.tvs_err_drive_enabled and agl is not None
                and p.tvs_err_drive_agl_low <= agl <= p.tvs_err_drive_agl_high):
            # Persistence check uses RAW v_norm: a single-frame spike drops
            # back next tick and resets the counter, so EMA residue cannot
            # accumulate into a false trigger.
            abs_err_raw = abs(v_norm)
            if abs_err_raw >= p.tvs_err_drive_deadband:
                self._tvs_err_confirm_count += 1
                if self._tvs_err_confirm_count >= p.tvs_err_drive_confirm_ticks:
                    # Magnitude uses the EMA: linear map deadband->saturate
                    abs_err_ema = abs(self._tvs_v_norm_ema)
                    span = max(p.tvs_err_drive_saturate
                               - p.tvs_err_drive_deadband, 1e-6)
                    frac = np.clip((abs_err_ema - p.tvs_err_drive_deadband) / span,
                                   0.0, 1.0)
                    err_floor = float(frac * p.tvs_err_drive_max_floor)
            else:
                self._tvs_err_confirm_count = 0
        else:
            self._tvs_err_confirm_count = 0
        self._tvs_err_floor = err_floor

        # beta = max(s schedule, error floor)
        beta_sched = max(beta_sched, err_floor)

        if beta_sched > 0.0:
            self._tvs_entered = True

        if self._tvs_entered and p.tvs_latch_once_entered:
            # Latch: beta monotone non-decreasing (proximity only grows;
            # a shrinking box is usually YOLO jitter)
            beta = max(beta_sched, self._tvs_beta)
        else:
            beta = beta_sched

        # Edge detection uses bbox edges, not the center
        bw_half = max(bbox_w, 0.0) / 2.0
        bh_half = max(bbox_h, 0.0) / 2.0
        m_x = p.tvs_edge_margin_frac * screen_w
        m_y = p.tvs_edge_margin_frac * screen_h
        edge = ((cx - bw_half) < m_x or (cx + bw_half) > (screen_w - m_x)
                or (cy - bh_half) < m_y or (cy + bh_half) > (screen_h - m_y))
        self._tvs_edge_flagged = bool(edge)
        if edge:
            # Center unreliable -> down-weight but keep some direction info
            beta *= p.tvs_edge_weight

        self._tvs_beta = float(np.clip(beta, 0.0, 1.0))

    def _apply_terminal_visual_servo(self, pitch_cmd_deg, heading_offset_deg):
        """
        Additive pure-P TVS correction on top of the MPC output, 100 Hz.
        Never written back into the MPC cache; has its own LPF + slew.

        Sign convention (valid for gimbal lock near (0,0)):
          u_norm > 0 -> bbox right of center -> heading right (offset +)
          v_norm > 0 -> bbox below center -> pitch more negative

        Returns:
            (pitch_cmd_deg_new, heading_offset_deg_new)
        """
        p = self.mpc_params

        if not p.tvs_enabled or self._tvs_beta <= 1e-4:
            # Not active -> smoothly decay the correction to zero
            self._tvs_corr_pitch_deg *= 0.5
            self._tvs_corr_heading_deg *= 0.5
            self._tvs_active = False
            return pitch_cmd_deg, heading_offset_deg

        beta = self._tvs_beta

        # Target correction (pure P)
        heading_corr = beta * p.tvs_K_lat_deg * self._tvs_u_norm
        pitch_corr   = -beta * p.tvs_K_lon_deg * self._tvs_v_norm

        # Clip to beta-scaled caps (authority grows with proximity)
        max_h = beta * p.tvs_max_corr_heading_deg
        max_p = beta * p.tvs_max_corr_pitch_deg
        heading_corr = float(np.clip(heading_corr, -max_h, max_h))
        pitch_corr   = float(np.clip(pitch_corr,   -max_p, max_p))

        # First-order LPF (raw bbox is noisy)
        a = p.tvs_corr_lpf_alpha
        heading_corr = a * self._tvs_corr_heading_deg + (1.0 - a) * heading_corr
        pitch_corr   = a * self._tvs_corr_pitch_deg   + (1.0 - a) * pitch_corr

        # Slew (hard rate limit)
        s = p.tvs_corr_slew_deg
        heading_corr = float(np.clip(
            heading_corr,
            self._tvs_corr_heading_deg - s, self._tvs_corr_heading_deg + s))
        pitch_corr = float(np.clip(
            pitch_corr,
            self._tvs_corr_pitch_deg - s, self._tvs_corr_pitch_deg + s))

        self._tvs_corr_heading_deg = heading_corr
        self._tvs_corr_pitch_deg = pitch_corr
        self._tvs_active = True

        return pitch_cmd_deg + pitch_corr, heading_offset_deg + heading_corr

    def get_tvs_debug(self):
        """TVS diagnostics dict for the main-loop data log."""
        return {
            'tvs_active':           1 if self._tvs_active else 0,
            'tvs_beta':             float(self._tvs_beta),
            'tvs_u_norm':           float(self._tvs_u_norm),
            'tvs_v_norm':           float(self._tvs_v_norm),
            'tvs_s_raw':            float(self._tvs_s_raw),
            'tvs_s_ema':            float(self._tvs_s_ema),
            'tvs_corr_pitch_deg':   float(self._tvs_corr_pitch_deg),
            'tvs_corr_heading_deg': float(self._tvs_corr_heading_deg),
            'tvs_detected':         1 if self._tvs_detected else 0,
            'tvs_edge_flagged':     1 if self._tvs_edge_flagged else 0,
            'tvs_lost_ticks':       int(self._tvs_lost_ticks),
            'tvs_entered':          1 if self._tvs_entered else 0,
            'tvs_beta_sched':       float(self._tvs_beta_sched),
            'tvs_err_floor':        float(self._tvs_err_floor),
            'tvs_v_norm_ema':       float(self._tvs_v_norm_ema),
        }

    # ════════════════════════════════════════════
    #  DIVE control
    # ════════════════════════════════════════════

    def dive_control(self, lateral_error, longitudinal_error,
                     current_pitch, current_roll, current_heading, current_r,
                     current_q, current_p, target_detected,
                     alpha_meas=None, gamma_true=None):
        """
        DIVE: 3DOF MPC -> (chi_cmd, gamma_cmd) -> (pitch_cmd, heading_cmd)
        -> cascade inner loop.
        """
        p = self.mpc_params

        self.update_dynamic_alpha_compensation(
            current_pitch, alpha_meas=alpha_meas, gamma_true=gamma_true)

        # 1. No target / no state -> fall back to cascade
        if not target_detected or self._latest_state is None:
            rudder, elevator, aileron, pitch_cmd_pre, yaw_cmd_pre = self.pre_dive_control(
                current_pitch, current_roll, current_heading, current_r,
                current_q=current_q, current_p=current_p,
                gimbal_pitch=self._latest_gimbal_pitch)
            rudder = self._apply_rudder_slew(rudder)
            aileron = self._apply_aileron_lpf(aileron)
            self._last_pitch_cmd_for_log = pitch_cmd_pre
            self._last_yaw_cmd_for_log = yaw_cmd_pre
            return rudder, elevator, aileron, pitch_cmd_pre, yaw_cmd_pre

        # 1.5 Terminal guard: image-based LoS, KF-independent.
        # The target KF can drift forward during a ~5 s DIVE; ray_world comes
        # straight from the image and always reflects the true direction.
        # Mid-phase uses MPC + KF estimate (prediction); terminal phase uses
        # pure visual LoS guidance.
        agl = (self._last_ac_altitude if self._last_ac_altitude is not None else 999.0)
        in_terminal = (p.terminal_guard_enabled and agl < p.terminal_guard_agl_m)
        if in_terminal:
            # Image-based LoS pitch command.
            # ray_world = (rx, ry, rz), NED z-down positive:
            #   pitch_LoS = -atan2(rz, sqrt(rx^2 + ry^2))
            pitch_cmd_los = p.dive_pitch_target_deg   # fallback
            terminal_mode = "fallback (no ray)"
            if self._last_ray_world is not None:
                rx, ry, rz = self._last_ray_world
                ray_horiz = math.sqrt(rx * rx + ry * ry)
                if ray_horiz > 1e-3:    # guard atan2 when target directly below
                    los_rad = -math.atan2(rz, ray_horiz)
                    los_deg = math.degrees(los_rad)
                    # pitch_cmd = LoS + alpha_nominal, clipped to MPC bounds
                    pitch_cmd_los = los_deg + p.alpha_nominal_deg
                    pitch_cmd_los = max(min(pitch_cmd_los, p.pitch_cmd_max_deg), p.pitch_cmd_min_deg)
                    terminal_mode = f"image LoS={los_deg:+.0f}, ray=({rx:+.2f},{ry:+.2f},{rz:+.2f})"

            # LPF + slew on the terminal pitch command; slew wide enough
            # (0.5/tick = 50 deg/s) to follow the true LoS rate near impact.
            if self._mpc_cached_pitch_cmd is not None:
                a = p.cmd_lpf_alpha
                pitch_cmd_los_lpf = a * self._mpc_cached_pitch_cmd + (1.0 - a) * pitch_cmd_los
                slew_per_tick = 0.5
                pitch_cmd_los_lpf = max(
                    min(pitch_cmd_los_lpf, self._mpc_cached_pitch_cmd + slew_per_tick),
                    self._mpc_cached_pitch_cmd - slew_per_tick)
                pitch_cmd_los = pitch_cmd_los_lpf
                self._mpc_cached_pitch_cmd = pitch_cmd_los

            pitch_cmd_locked = pitch_cmd_los
            heading_cmd_locked = (self._mpc_cached_heading_cmd
                if self._mpc_cached_heading_cmd is not None
                else current_heading)
            # TVS: the terminal heading would otherwise be frozen (no lateral
            # closed loop); the measured bbox closes it here.
            pitch_cmd_locked, _tvs_h_corr = self._apply_terminal_visual_servo(
                pitch_cmd_locked, 0.0)
            heading_cmd_locked = heading_cmd_locked + _tvs_h_corr
            if (p.tvs_enabled and self._tvs_active
                    and p.tvs_terminal_heading_dev_deg is not None):
                _max_dev = p.tvs_terminal_heading_dev_deg
                heading_cmd_locked = float(np.clip(
                    heading_cmd_locked,
                    current_heading - _max_dev, current_heading + _max_dev))
            rudder, elevator, aileron = self._v11_inner_loop(
                pitch_cmd=pitch_cmd_locked, heading_cmd=heading_cmd_locked,
                current_pitch=current_pitch, current_roll=current_roll,
                current_heading=current_heading, current_r=current_r,
                current_q=current_q, current_p=current_p)
            self._last_pitch_cmd_for_log = pitch_cmd_locked
            self._last_yaw_cmd_for_log = heading_cmd_locked
            if not self._terminal_guard_announced:
                print(f"  [MPC] terminal guard active (AGL={agl:.0f}m < "
                      f"{p.terminal_guard_agl_m}m), pcmd={pitch_cmd_locked:+.1f} ({terminal_mode})")
                self._terminal_guard_announced = True
            return rudder, elevator, aileron, pitch_cmd_locked, heading_cmd_locked

        # 2. Low-rate MPC solve
        # Init guard: ensure cache exists before the first solve
        if self._mpc_cached_pitch_cmd is None:
            self._mpc_cached_pitch_cmd = p.dive_pitch_target_deg
            self._mpc_cached_heading_offset_deg = 0.0
            self._mpc_cached_heading_cmd = current_heading
            if self._dive_start_time is None:
                self._dive_start_time = time.time()
            print(f"  [MPC] dive_control first run, cache auto-initialized")

        self._mpc_step_counter += 1
        should_solve = (
            self._mpc_step_counter >= p.mpc_solve_interval_ticks
            or self._mpc_cached_pitch_cmd is None
        )

        if should_solve:
            try:
                pitch_cmd_raw, heading_offset_deg = self._mpc_solve_for_setpoint(
                    current_pitch, current_heading)
                self._mpc_step_counter = 0
            except Exception as e:
                print(f"  [MPC] solve failed: {e}, using cached cmd / cascade")
                pitch_cmd_raw = self._mpc_cached_pitch_cmd
                heading_offset_deg = self._mpc_cached_heading_offset_deg

            if pitch_cmd_raw is not None:
                if not (np.isfinite(pitch_cmd_raw) and np.isfinite(heading_offset_deg)):
                    print(f"  [MPC] NaN/Inf detected, fallback")
                    pitch_cmd_raw = None

            if pitch_cmd_raw is not None:
                # Absolute bounds
                pitch_cmd_raw = float(np.clip(
                    pitch_cmd_raw, p.pitch_cmd_min_deg, p.pitch_cmd_max_deg))
                heading_offset_deg = float(np.clip(
                    heading_offset_deg,
                    -p.heading_cmd_max_dev_deg, p.heading_cmd_max_dev_deg))

                # LPF (before slew): smooths solver jitter
                if self._mpc_cached_pitch_cmd is not None:
                    a = p.cmd_lpf_alpha
                    pitch_cmd_raw = a * self._mpc_cached_pitch_cmd + (1.0 - a) * pitch_cmd_raw
                    heading_offset_deg = a * self._mpc_cached_heading_offset_deg + (1.0 - a) * heading_offset_deg

                # Slew (hard limit after LPF)
                if self._mpc_cached_pitch_cmd is not None:
                    pitch_cmd_raw = float(np.clip(
                        pitch_cmd_raw,
                        self._mpc_cached_pitch_cmd - p.cmd_slew_pitch_deg,
                        self._mpc_cached_pitch_cmd + p.cmd_slew_pitch_deg))
                    heading_offset_deg = float(np.clip(
                        heading_offset_deg,
                        self._mpc_cached_heading_offset_deg - p.cmd_slew_heading_deg,
                        self._mpc_cached_heading_offset_deg + p.cmd_slew_heading_deg))
                # DIVE-entry anti-pull-up guard: first 0.3 s the command may
                # not be shallower than dive_pitch_target
                if self._dive_start_time is not None:
                    dive_elapsed = time.time() - self._dive_start_time
                    if dive_elapsed < 0.3:
                        pitch_cmd_raw = min(pitch_cmd_raw, p.dive_pitch_target_deg)
                self._mpc_cached_pitch_cmd = pitch_cmd_raw
                self._mpc_cached_heading_offset_deg = heading_offset_deg
                # Lock the absolute heading target at solve time so the
                # inner loop sees a real tracking error between solves
                self._mpc_cached_heading_cmd = current_heading + heading_offset_deg

                self._mpc_solve_count += 1
                if self._mpc_solve_count % 5 == 0:
                    avg_ms = (np.mean(list(self._mpc_solve_times)) * 1000
                              if self._mpc_solve_times else 0)
                    info = self._mpc_last_info
                    extra = ""
                    if info is not None:
                        extra = (f" gamma_cmd={info['gamma_cmd_deg']:+.1f} "
                                 f"chi_off={info['chi_cmd_offset_deg']:+.1f}")
                    print(f"  [MPC] {avg_ms:.0f}ms "
                          f"pitch_cmd={pitch_cmd_raw:+.1f} (theta={current_pitch:+.1f}) "
                          f"hdg_off={heading_offset_deg:+.2f}{extra}")

        # 3. Cache still empty -> cascade fallback
        if self._mpc_cached_pitch_cmd is None:
            rudder, elevator, aileron, pitch_cmd_pre, yaw_cmd_pre = self.pre_dive_control(
                current_pitch, current_roll, current_heading, current_r,
                current_q=current_q, current_p=current_p,
                gimbal_pitch=self._latest_gimbal_pitch)
            rudder = self._apply_rudder_slew(rudder)
            aileron = self._apply_aileron_lpf(aileron)
            self._last_pitch_cmd_for_log = pitch_cmd_pre
            self._last_yaw_cmd_for_log = yaw_cmd_pre
            return rudder, elevator, aileron, pitch_cmd_pre, yaw_cmd_pre

        # 4. Use cached command
        pitch_cmd = self._mpc_cached_pitch_cmd
        heading_cmd = (self._mpc_cached_heading_cmd
                       if self._mpc_cached_heading_cmd is not None
                       else current_heading)

        # 4.5 TVS overlay (100 Hz, not written back into the cache;
        # beta=0 makes this a no-op)
        pitch_cmd, _tvs_h_corr = self._apply_terminal_visual_servo(pitch_cmd, 0.0)
        heading_cmd = heading_cmd + _tvs_h_corr
        # Protective clip on the total heading command (relative to current)
        if (p.tvs_enabled and self._tvs_active
                and p.tvs_terminal_heading_dev_deg is not None):
            _max_dev = p.tvs_terminal_heading_dev_deg
            heading_cmd = float(np.clip(heading_cmd,
                                        current_heading - _max_dev,
                                        current_heading + _max_dev))

        # 5. Inner loop
        rudder, elevator, aileron = self._v11_inner_loop(
            pitch_cmd=pitch_cmd, heading_cmd=heading_cmd,
            current_pitch=current_pitch, current_roll=current_roll,
            current_heading=current_heading, current_r=current_r,
            current_q=current_q, current_p=current_p)

        self._last_pitch_cmd_for_log = pitch_cmd
        self._last_yaw_cmd_for_log = heading_cmd
        return rudder, elevator, aileron, pitch_cmd, heading_cmd

    # ════════════════════════════════════════════
    #  MPC solve -> setpoint conversion
    # ════════════════════════════════════════════

    def _mpc_solve_for_setpoint(self, current_pitch, current_heading):
        """
        Assemble (Dx, Dy, h, V, chi, gamma) from sensors, solve, output
        (pitch_cmd_deg, heading_offset_deg).

        Sources: Dx/Dy from target_kf minus aircraft NED; h from AGL;
        V from airspeed; chi from NED velocity; gamma from X-Plane vpath.
        """
        p = self.mpc_params
        state = self._latest_state

        # 1. V (m/s)
        airspeed_kt = float(state[0])
        V = airspeed_kt * 0.51444
        if V < 5.0:
            V = 5.0

        # 2. chi (track angle, rad) from NED velocity
        vx_n = self._last_ac_vx_ned
        vy_n = self._last_ac_vy_ned
        if abs(vx_n) < 0.1 and abs(vy_n) < 0.1:
            chi = self._last_ac_heading_rad   # fallback at low speed
        else:
            chi = math.atan2(vy_n, vx_n)

        # 3. gamma (flight-path angle, rad) from X-Plane vpath
        gamma_deg = float(state[15])
        gamma = math.radians(gamma_deg)

        # 4. h (AGL, m)
        h = (self._last_ac_altitude if self._last_ac_altitude is not None
             else 200.0)
        if h < 1.0:
            h = 1.0

        # 5. Dx, Dy (aircraft relative to target, NED)
        if not self.target_kf._initialized:
            print(f"  [MPC] target_kf not initialized, fallback")
            return None, None

        target_x_ned = float(self.target_kf.x[0])
        target_y_ned = float(self.target_kf.x[1])

        # Fixed lead-time velocity feed-forward, scaled by KF confidence;
        # the image cost does the active correction
        if p.use_velocity_feedforward:
            target_vx, target_vy = self.target_kf.get_velocity()
            confidence = self.target_kf.get_confidence(
                full_std=p.kf_vel_std_full_confidence,
                zero_std=p.kf_vel_std_zero_confidence)
            ff_lead = p.lead_time * confidence
            target_x_lead = target_x_ned + target_vx * ff_lead
            target_y_lead = target_y_ned + target_vy * ff_lead
        else:
            target_x_lead = target_x_ned
            target_y_lead = target_y_ned

        ac_x = self._last_ac_x_ned
        ac_y = self._last_ac_y_ned

        Dx = ac_x - target_x_lead
        Dy = ac_y - target_y_lead

        # 6. Assemble x0 (absolute chi; solver converts to offset)
        x0 = np.array([Dx, Dy, h, V, chi, gamma])

        if self._mpc_solve_count < 3:
            print(f"  [MPC DEBUG] solve#{self._mpc_solve_count}: "
                  f"ac_NED=({ac_x:+.0f},{ac_y:+.0f}) "
                  f"target_NED=({target_x_ned:+.0f},{target_y_ned:+.0f}) "
                  f"D=({Dx:+.0f},{Dy:+.0f}) "
                  f"h={h:.0f} V={V:.1f} chi={math.degrees(chi):+.1f} "
                  f"gamma={math.degrees(gamma):+.1f}")

        # 7. Solve (chi_initial = current chi; u_prev = last absolute commands)
        chi_cmd_rad, gamma_cmd_rad, info = self.mpc_solver.solve(
            x0_full=x0,
            R_b2c=self.R_b2c_locked,
            u_prev_full=self._mpc_u_prev,
            chi_initial=chi,
            w_gamma_anchor_eff=p.w_gamma_anchor,
            w_image_y_eff=p.w_image_y,
        )

        # 8. Convert to inner-loop commands
        pitch_cmd_deg = math.degrees(gamma_cmd_rad) + p.alpha_nominal_deg
        heading_offset_deg = info['chi_cmd_offset_deg']

        self._mpc_u_prev = info['u_first_absolute'].copy()
        self._mpc_solve_times.append(info['solve_time'])
        self._mpc_last_info = info

        return pitch_cmd_deg, heading_offset_deg

    # ════════════════════════════════════════════
    #  Cascade inner loop
    # ════════════════════════════════════════════

    def _v11_inner_loop(self, pitch_cmd, heading_cmd,
                         current_pitch, current_roll, current_heading,
                         current_r, current_q, current_p):
        return compute_inner_loop(
            self, pitch_cmd, heading_cmd, current_pitch, current_roll,
            current_heading, current_r, current_q, current_p)

    # ──────────────────────────────────────────
    #  Output shaping helpers
    # ──────────────────────────────────────────

    def _apply_rudder_slew(self, rudder):
        return apply_rudder_slew(self, rudder)

    def _apply_aileron_lpf(self, aileron):
        return apply_aileron_lpf(self, aileron)

    def reset_controllers(self):
        """Reset all MPC + base-class state."""
        super().reset_controllers()
        self.mpc_solver.reset()
        self.drift_estimator.reset()
        self.target_kf.reset() if hasattr(self.target_kf, 'reset') else None
        self.depth_ema.reset()

        self._mpc_u_prev = np.zeros(2)
        self._mpc_solve_times.clear()
        self._mpc_solve_count = 0
        self._mpc_last_info = None

        self._mpc_cached_pitch_cmd = None
        self._mpc_cached_heading_offset_deg = 0.0
        self._mpc_cached_heading_cmd = None
        self._mpc_step_counter = self.mpc_params.mpc_solve_interval_ticks

        self._dive_start_time = None
        self.gimbal_pitch_locked = 0.0
        self.gimbal_yaw_locked = 0.0
        self.R_b2c_locked = compute_R_b2c(0.0, 0.0)

        self._last_slant_range = None
        self._last_ground_range = None
        self._last_ac_altitude = None
        self._last_ac_x_ned = 0.0
        self._last_ac_y_ned = 0.0
        self._world_state_update_count = 0
        self._dkf_drift_lon = 0.0
        self._dkf_drift_lat = 0.0
        self._dkf_drift_valid = False
        self._terminal_guard_announced = False
        self._kf_rejected_count = 0
        self._kf_dive_reset_pending = False
        self._kf_dive_reset_done_count = 0
        self._prev_rudder_v22 = 0.0
        self._rudder_lpf_state = 0.0
        self._aileron_lpf_state = 0.0

        self._manual_initial_heading = None
        self._manual_initial_alt = None
        self._manual_dive_armed = False

        # TVS state
        self._tvs_u_norm = 0.0
        self._tvs_v_norm = 0.0
        self._tvs_s_raw = 0.0
        self._tvs_s_ema = 0.0
        self._tvs_beta = 0.0
        self._tvs_entered = False
        self._tvs_detected = False
        self._tvs_edge_flagged = False
        self._tvs_lost_ticks = 0
        self._tvs_corr_pitch_deg = 0.0
        self._tvs_corr_heading_deg = 0.0
        self._tvs_active = False
        self._tvs_last_update_count = 0
        self._tvs_v_norm_ema = 0.0
        self._tvs_err_confirm_count = 0
        self._tvs_err_floor = 0.0
        self._tvs_beta_sched = 0.0

        print("[MPC] reset")
