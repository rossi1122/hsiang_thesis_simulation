"""
Ego-aware delayed Kalman filter for pixel-space target tracking,
plus NED <-> pixel reprojection utilities.
"""

import math
import numpy as np
from collections import deque


# ════════════════════════════════════════════════════════════════════
#  Rotation utilities
# ════════════════════════════════════════════════════════════════════

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


# Gimbal -> Camera (CV convention: X right, Y down, Z forward)
DEFAULT_R_G2C = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
], dtype=float)


# ════════════════════════════════════════════════════════════════════
#  NED -> pixel reprojection
# ════════════════════════════════════════════════════════════════════

def reproject_target_ned_to_pixel(
    target_x_ned, target_y_ned,
    ac_x_ned, ac_y_ned, ac_altitude_agl,
    ac_roll_deg, ac_pitch_deg, ac_yaw_true_deg,
    gimbal_pitch_deg, gimbal_yaw_deg,
    fx, fy, cx, cy,
    R_g2c=None,
    gimbal_pitch_offset_deg=0.0,
    target_altitude_msl=0.0,
):
    """
    Project a NED target position into camera pixel coordinates.

    Returns:
        (u, v, valid). valid=False if the target is behind the camera.
    """
    if R_g2c is None:
        R_g2c = DEFAULT_R_G2C

    # Vector aircraft -> target in NED (down positive)
    dx = target_x_ned - ac_x_ned          # north
    dy = target_y_ned - ac_y_ned          # east
    dz = ac_altitude_agl - target_altitude_msl  # down
    vec_ned = np.array([dx, dy, dz], dtype=float)

    # Rotation chain: NED -> Body -> Gimbal -> Camera (CV)
    R_w2b = _Rx(ac_roll_deg) @ _Ry(ac_pitch_deg) @ _Rz(ac_yaw_true_deg)
    gimbal_pitch_eff = gimbal_pitch_deg - gimbal_pitch_offset_deg
    R_b2g = _Ry(gimbal_pitch_eff) @ _Rz(gimbal_yaw_deg)
    R_ned2cam = R_g2c @ R_b2g @ R_w2b

    Pc = R_ned2cam @ vec_ned

    if Pc[2] <= 0.1:
        return None, None, False

    # Pinhole projection
    u = fx * Pc[0] / Pc[2] + cx
    v = fy * Pc[1] / Pc[2] + cy

    return float(u), float(v), True


# ════════════════════════════════════════════════════════════════════
#  Ego-aware DKF
# ════════════════════════════════════════════════════════════════════

class EgoAwareDKF:
    """
    Delayed KF with ego-motion compensation.

    State: [u, v, u_dot, v_dot] (pixel position + residual pixel velocity).
    Ego motion (aircraft + gimbal) is removed in predict() via projective
    compensation, so [u_dot, v_dot] carries only target-induced motion.

    API:
      - predict(ac_state, gimbal_state, depth_m): high-rate prediction
      - update(z, delay_steps, bbox_size): delayed YOLO measurement
      - update_fallback(z, r_std_pixel): NED-reprojection measurement
    """

    def __init__(self,
                 dt=0.01,
                 image_delay=0.08,
                 fx=1500.0, fy=1500.0, cx=960.0, cy=505.0,
                 enable_ego_comp=True,
                 max_history_steps=200,
                 max_update_delay_steps=30):
        """
        Args:
            max_history_steps: history buffer size (steps); must cover max delay.
            max_update_delay_steps: cap on delay compensation depth; larger
                delays should be handled by the NED fallback.
        """
        self.dt = dt
        self.image_delay = image_delay
        self.default_delay_steps = max(1, int(image_delay / dt))
        self.max_update_delay_steps = max_update_delay_steps

        # Camera intrinsics
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.enable_ego_comp = enable_ego_comp

        # State
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0

        # Constant-velocity propagation on residual velocity
        self.F_base = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])

        # Process noise for residual velocity only (ego motion removed)
        accel_noise = 20.0  # px/s^2
        dt2 = dt * dt
        self.Q = (accel_noise ** 2) * np.array([
            [dt2*dt2/4, 0,         dt2*dt/2, 0        ],
            [0,         dt2*dt2/4, 0,        dt2*dt/2 ],
            [dt2*dt/2,  0,         dt2,      0        ],
            [0,         dt2*dt/2,  0,        dt2      ],
        ])

        # Dynamic measurement noise (scaled by bbox size)
        self.R_base_std = 8.0
        self.R_bbox_ref = 32.0
        self.R_min_std = 5.0    # floor prevents over-tight gate at close range
        self.R_max_std = 25.0

        # Extra prediction noise from ego compensation errors
        self.ego_uncertainty_std = 5.0  # px/s

        # Mahalanobis gates
        self.gate_threshold = 200.0
        self.gate_threshold_fallback = 200.0

        max_history = max(self.default_delay_steps + 30, max_history_steps)
        self._history = deque(maxlen=max_history)
        self._max_history_size = max_history

        self.initialized = False
        self._predict_count = 0
        self._update_count = 0
        self._update_fallback_count = 0
        self._reject_count = 0

        # Coasting mode (mild: ego comp already removes most drift)
        self._consecutive_miss = 0
        self.coast_start = 25
        self.coast_q_inflation = 1.5
        self.coast_q_rate = 0.03
        self.coast_vel_decay = 0.999
        self.coast_vel_decay_fast = 0.99

        self._last_ego_dpx = np.zeros(2)
        self._last_ego_uncertainty_var = 0.0

        # Previous aircraft/gimbal state for projective ego shift
        self._prev_ac_state = None
        self._prev_gimbal_state = None

        print(f"[DKF] init:")
        print(f"  predict rate: {1/dt:.0f} Hz")
        print(f"  default delay: {image_delay*1000:.0f} ms ({self.default_delay_steps} steps)")
        print(f"  history buffer: {max_history} steps = {max_history*dt*1000:.0f} ms")
        print(f"  update delay cap: {max_update_delay_steps} steps = "
              f"{max_update_delay_steps*dt*1000:.0f} ms (beyond -> NED fallback)")
        print(f"  ego comp: {'ON' if enable_ego_comp else 'OFF'}")
        print(f"  intrinsics: fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f}")
        print(f"  process noise (residual): {accel_noise:.0f} px/s^2")
        print(f"  coasting: start={self.coast_start} steps "
              f"Q_inflate={self.coast_q_inflation}x")

    # ─────────────────────────────────────────────────────────────
    #  Predict
    # ─────────────────────────────────────────────────────────────

    def predict(self,
                ac_state=None,         # dict: x_ned, y_ned, alt_agl, roll_deg, pitch_deg, yaw_true_deg
                gimbal_state=None,     # dict: pitch_deg, yaw_deg
                depth_m=None):
        """
        High-rate prediction with projective ego compensation.

        Back-project the current pixel with depth to a NED point (previous
        pose), re-project with the current pose; the difference is the
        ego-motion pixel shift (aircraft + gimbal included automatically).

        Calling conventions:
          predict()                              -> pure CV, no ego comp
          predict(ac_state, gimbal_state)        -> uses last depth if any
          predict(ac_state, gimbal_state, depth) -> full ego comp (recommended)
        """
        if not self.initialized:
            return

        self._consecutive_miss += 1

        # Coasting
        if self._consecutive_miss > self.coast_start:
            coast_steps = self._consecutive_miss - self.coast_start
            q_factor = min(1.0 + coast_steps * self.coast_q_rate, self.coast_q_inflation)
            Q_eff = self.Q * q_factor
            decay = self.coast_vel_decay if coast_steps < 30 else self.coast_vel_decay_fast
            self.x[2] *= decay
            self.x[3] *= decay
        else:
            Q_eff = self.Q

        # Projective ego compensation
        ego_dpx = np.zeros(2)
        ego_var = 0.0

        if (self.enable_ego_comp
                and ac_state is not None
                and gimbal_state is not None
                and self._prev_ac_state is not None
                and self._prev_gimbal_state is not None
                and depth_m is not None and depth_m > 1.0):
            # Detect gimbal angle jumps (e.g. lock/unlock); skip ego comp this step
            d_gimbal_pitch = abs(gimbal_state['pitch_deg'] - self._prev_gimbal_state['pitch_deg'])
            d_gimbal_yaw   = abs(gimbal_state['yaw_deg']   - self._prev_gimbal_state['yaw_deg'])
            gimbal_jumped = (d_gimbal_pitch > 1.5) or (d_gimbal_yaw > 1.5)

            if not gimbal_jumped:
                # Back-project with previous pose
                P_ned = self._backproject_pixel_to_ned(
                    u_pixel=self.x[0], v_pixel=self.x[1],
                    depth=depth_m,
                    ac_state=self._prev_ac_state,
                    gimbal_state=self._prev_gimbal_state,
                )
                # Re-project with current pose
                u_new, v_new, valid = reproject_target_ned_to_pixel(
                    target_x_ned=P_ned[0],
                    target_y_ned=P_ned[1],
                    ac_x_ned=ac_state['x_ned'],
                    ac_y_ned=ac_state['y_ned'],
                    ac_altitude_agl=ac_state['alt_agl'],
                    ac_roll_deg=ac_state['roll_deg'],
                    ac_pitch_deg=ac_state['pitch_deg'],
                    ac_yaw_true_deg=ac_state['yaw_true_deg'],
                    gimbal_pitch_deg=gimbal_state['pitch_deg'],
                    gimbal_yaw_deg=gimbal_state['yaw_deg'],
                    fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                    R_g2c=DEFAULT_R_G2C,
                    target_altitude_msl=P_ned[2],
                )
                if valid:
                    raw_dpx_x = u_new - self.x[0]
                    raw_dpx_y = v_new - self.x[1]
                    raw_dpx_norm = math.sqrt(raw_dpx_x**2 + raw_dpx_y**2)

                    # Clip abnormally large single-step shifts (bad depth/attitude)
                    EGO_DPX_MAX_PIXEL = 30.0
                    if raw_dpx_norm > EGO_DPX_MAX_PIXEL:
                        scale = EGO_DPX_MAX_PIXEL / raw_dpx_norm
                        ego_dpx[0] = raw_dpx_x * scale
                        ego_dpx[1] = raw_dpx_y * scale
                        ego_var = (raw_dpx_norm * 0.5) ** 2 + 100.0
                    else:
                        ego_dpx[0] = raw_dpx_x
                        ego_dpx[1] = raw_dpx_y
                        ego_var = (0.05 * raw_dpx_norm) ** 2 + 1.0
            else:
                if not hasattr(self, '_jump_count'):
                    self._jump_count = 0
                self._jump_count += 1
                if self._jump_count % 10 == 1:
                    print(f"[DKF] gimbal jump detected "
                          f"(d_pitch={d_gimbal_pitch:.2f} d_yaw={d_gimbal_yaw:.2f}), "
                          f"skip ego comp [count={self._jump_count}]")

        # Apply: CV propagation first, then ego shift on position
        self.x = self.F_base @ self.x
        self.x[0] += ego_dpx[0]
        self.x[1] += ego_dpx[1]

        self.P = self.F_base @ self.P @ self.F_base.T + Q_eff
        self.P[0, 0] += ego_var
        self.P[1, 1] += ego_var

        self._last_ego_dpx = ego_dpx
        self._last_ego_uncertainty_var = ego_var
        if ac_state is not None:
            self._prev_ac_state = dict(ac_state)
        if gimbal_state is not None:
            self._prev_gimbal_state = dict(gimbal_state)

        self._predict_count += 1
        self._history.append((self.x.copy(), self.P.copy(), ego_dpx.copy()))

    def _backproject_pixel_to_ned(self, u_pixel, v_pixel, depth,
                                   ac_state, gimbal_state):
        """Back-project (pixel + depth) into an absolute NED point."""
        # Pixel -> camera ray
        x1 = (u_pixel - self.cx) / self.fx
        x2 = (v_pixel - self.cy) / self.fy
        Pc_dir = np.array([x1, x2, 1.0])
        Pc = depth * Pc_dir / np.linalg.norm(Pc_dir)

        # Camera -> NED (inverse of NED -> Body -> Gimbal -> Camera)
        R_w2b = (_Rx(ac_state['roll_deg'])
                 @ _Ry(ac_state['pitch_deg'])
                 @ _Rz(ac_state['yaw_true_deg']))
        R_b2g = (_Ry(gimbal_state['pitch_deg'])
                 @ _Rz(gimbal_state['yaw_deg']))
        R_ned2cam = DEFAULT_R_G2C @ R_b2g @ R_w2b
        R_cam2ned = R_ned2cam.T

        rel_ned = R_cam2ned @ Pc

        ac_ned = np.array([ac_state['x_ned'],
                            ac_state['y_ned'],
                            -ac_state['alt_agl']])
        return ac_ned + rel_ned

    # ─────────────────────────────────────────────────────────────
    #  Update (delayed YOLO measurement)
    # ─────────────────────────────────────────────────────────────

    def update(self, z_image, delay_steps=None, bbox_size=None):
        """Delayed YOLO measurement update with history-based compensation."""
        z = np.asarray(z_image, dtype=float)

        if not self.initialized:
            self.x[0:2] = z
            self.x[2:4] = 0.0
            self.P = np.diag([20.0**2, 20.0**2, 50.0**2, 50.0**2])
            self.initialized = True
            self._history.append((self.x.copy(), self.P.copy(), np.zeros(2)))
            return

        # Dynamic R from bbox size
        if bbox_size is not None and bbox_size > 5:
            r_std = self.R_base_std * (self.R_bbox_ref / bbox_size)
            r_std = np.clip(r_std, self.R_min_std, self.R_max_std)
        else:
            r_std = self.R_base_std
        R = np.diag([r_std**2, r_std**2])

        if delay_steps is None:
            delay_steps = self.default_delay_steps
        delay_steps = min(delay_steps, len(self._history) - 1)
        delay_steps = max(delay_steps, 0)
        # Cap compensation depth; long forward propagation amplifies
        # velocity-correction error. Larger delays -> NED fallback.
        delay_steps = min(delay_steps, self.max_update_delay_steps)

        # State at measurement time
        if delay_steps > 0 and len(self._history) > delay_steps:
            x_d, P_d, _ = self._history[-(delay_steps + 1)]
            x_d, P_d = x_d.copy(), P_d.copy()
        else:
            x_d, P_d = self.x.copy(), self.P.copy()
            delay_steps = 0

        # Innovation + gate
        y_innov = z - self.H @ x_d
        S = self.H @ P_d @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        mahal2 = float(y_innov.T @ S_inv @ y_innov)
        if mahal2 > self.gate_threshold:
            self._reject_count += 1
            if self._reject_count % 10 == 1:
                print(f"[DKF] YOLO measurement rejected (mahal2={mahal2:.1f}) "
                      f"z={z.astype(int)} vs pred={(self.H @ x_d).astype(int)}")
            return

        # KF update at delayed time
        K = P_d @ self.H.T @ S_inv
        x_corrected = x_d + K @ y_innov
        P_corrected = (np.eye(4) - K @ self.H) @ P_d

        # Forward-propagate with F_base only (history ego shifts would mismatch
        # after the correction; long delays rely on NED fallback)
        for _ in range(delay_steps):
            x_corrected = self.F_base @ x_corrected
            P_corrected = self.F_base @ P_corrected @ self.F_base.T + self.Q

        self.x = x_corrected
        self.P = P_corrected
        self._update_count += 1
        self._consecutive_miss = 0
        if self._history:
            _, _, last_ego = self._history[-1]
            self._history[-1] = (self.x.copy(), self.P.copy(), last_ego)

    # ─────────────────────────────────────────────────────────────
    #  Update (NED reprojection fallback)
    # ─────────────────────────────────────────────────────────────

    def update_fallback(self, z_pixel, r_std_pixel, gate_override=None):
        """
        NED reprojection fallback update.

        Differences from update(): external R, no delay compensation
        (reprojection is a current-time prediction), wider gate, and
        _consecutive_miss is not reset (not a real visual measurement).
        """
        z = np.asarray(z_pixel, dtype=float)

        if not self.initialized:
            self.x[0:2] = z
            self.x[2:4] = 0.0
            self.P = np.diag([r_std_pixel**2, r_std_pixel**2,
                              100.0**2, 100.0**2])
            self.initialized = True
            self._history.append((self.x.copy(), self.P.copy(), np.zeros(2)))
            return

        R = np.diag([r_std_pixel**2, r_std_pixel**2])
        gate = gate_override if gate_override is not None else self.gate_threshold_fallback

        y_innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        mahal2 = float(y_innov.T @ S_inv @ y_innov)
        if mahal2 > gate:
            self._reject_count += 1
            if self._reject_count % 20 == 1:
                print(f"[DKF] NED fallback rejected (mahal2={mahal2:.1f}, gate={gate:.0f})")
            return

        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self._update_fallback_count += 1
        if self._history:
            _, _, last_ego = self._history[-1]
            self._history[-1] = (self.x.copy(), self.P.copy(), last_ego)

    # ─────────────────────────────────────────────────────────────
    #  Getters
    # ─────────────────────────────────────────────────────────────

    def get_position(self):
        return self.x[0:2].copy()

    def get_velocity(self):
        """Residual pixel velocity (px/s, ego motion excluded)."""
        return self.x[2:4].copy()

    def get_position_std(self):
        return np.sqrt(np.diag(self.P)[:2])

    def get_stats(self):
        return {
            'predict_count': self._predict_count,
            'update_count': self._update_count,
            'update_fallback_count': self._update_fallback_count,
            'reject_count': self._reject_count,
            'consecutive_miss': self._consecutive_miss,
            'last_ego_dpx': self._last_ego_dpx.copy(),
        }

    def reset(self, initial_position=None):
        if initial_position is not None:
            self.x[0:2] = initial_position
            self.initialized = True
        else:
            self.x = np.zeros(4)
            self.initialized = False
        self.x[2:4] = 0.0
        self.P = np.eye(4) * 100.0
        self._history.clear()
        self._predict_count = 0
        self._update_count = 0
        self._update_fallback_count = 0
        self._reject_count = 0
        self._consecutive_miss = 0
        self._prev_ac_state = None
        self._prev_gimbal_state = None
