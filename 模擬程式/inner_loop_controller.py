"""Shared DIVE inner loop for PID / MPC controllers.
Tune the default values in DiveInnerLoopParams.

Angles: deg; angular rates: deg/s; control surface outputs: normalized.
Slew limits specify the maximum change per control tick.
The current control tick in main.py is 0.01 s.

Restart the program after changing parameters.
PRE_DIVE and fallback paths that explicitly call pre_dive_control
continue to use the original PRE_DIVE parameters.
Normal DIVE control and manual inner-loop tests use this module.

Filter states remain on the parent controller to preserve
PRE_DIVE-to-DIVE transitions and existing reset behavior.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DiveInnerLoopParams:
    # ===== Pitch: pitch angle -> q command -> elevator =====
    K_theta: float = 3.0
    K_q: float = 0.06
    max_q_cmd: float = 25.0
    pitch_cmd_min_deg: float = -60.0
    pitch_cmd_max_deg: float = 5.0
    elevator_limit: float = 1.0
    elevator_slew: float = 0.20
    elevator_lpf_alpha: float = 0.7  # 1.0 disables filtering

    # ===== Yaw: heading error and r damping -> rudder =====
    K_heading: float = 0.025
    K_r: float = 0.04
    rudder_limit: float = 0.20
    rudder_slew: float = 0.025
    rudder_lpf_alpha: float = 1.0

    # ===== Roll: hold zero roll angle with p damping -> aileron =====
    K_roll: float = 0.05
    K_p: float = 0.015
    aileron_limit: float = 0.25
    aileron_lpf_alpha: float = 0.5


def init_dive_inner_loop(controller):
    controller.dive_inner_params = DiveInnerLoopParams()
    # Preserve compatibility with existing diagnostic output.
    # Control calculations read dive_inner_params directly.
    controller.rudder_slew_v22 = controller.dive_inner_params.rudder_slew
    controller.rudder_lpf_alpha = controller.dive_inner_params.rudder_lpf_alpha
    controller.aileron_lpf_alpha = controller.dive_inner_params.aileron_lpf_alpha
    controller._prev_rudder_v22 = 0.0
    controller._rudder_lpf_state = 0.0
    controller._aileron_lpf_state = 0.0


def compute_inner_loop(self, pitch_cmd, heading_cmd,
                       current_pitch, current_roll, current_heading,
                       current_r, current_q, current_p):
    """
    Cascade inner loop.
    Do not filter q here: additional phase lag in the theta-q cascade
    can turn damping into anti-damping and cause self-excited oscillations.
    """
    p = self.dive_inner_params

    # Pitch: cascaded pitch-angle and pitch-rate control
    pitch_cmd_clip = float(np.clip(pitch_cmd, p.pitch_cmd_min_deg, p.pitch_cmd_max_deg))
    pitch_error = pitch_cmd_clip - current_pitch
    q_cmd = p.K_theta * pitch_error
    q_cmd = float(np.clip(q_cmd, -p.max_q_cmd, p.max_q_cmd))
    q_error = q_cmd - current_q
    elevator = p.K_q * q_error

    max_step = p.elevator_slew
    elevator = float(np.clip(elevator,
                             self.prev_pre_dive_elevator - max_step,
                             self.prev_pre_dive_elevator + max_step))
    elevator = float(np.clip(elevator, -p.elevator_limit, p.elevator_limit))

    # Elevator output LPF; filter state is shared with the base class
    a = p.elevator_lpf_alpha
    elevator = a * elevator + (1.0 - a) * self._elevator_lpf_state
    self._elevator_lpf_state = elevator

    self.prev_pre_dive_elevator = elevator

    heading_error = heading_cmd - current_heading
    if heading_error > 180:
        heading_error -= 360
    elif heading_error < -180:
        heading_error += 360

    # The original yaw-damper gain is 0.04.
    # Higher gains were noted to excite a 7-14 Hz limit cycle
    # at dive dynamic pressure.
    rudder = p.K_heading * heading_error - p.K_r * current_r
    rudder = float(np.clip(rudder, -p.rudder_limit, p.rudder_limit))

    rudder = self._apply_rudder_slew(rudder)

    aileron = -p.K_roll * current_roll - p.K_p * current_p
    aileron = float(np.clip(aileron, -p.aileron_limit, p.aileron_limit))

    aileron = self._apply_aileron_lpf(aileron)

    return rudder, elevator, aileron


def apply_rudder_slew(self, rudder):
    """
    Apply a rudder slew limit and optional output LPF.

    Apply this on every dive_control return path, including cascade
    fallbacks, to prevent sudden rudder changes during target dropout.
    """
    rudder = float(np.clip(rudder,
                           self._prev_rudder_v22 - self.dive_inner_params.rudder_slew,
                           self._prev_rudder_v22 + self.dive_inner_params.rudder_slew))
    self._prev_rudder_v22 = rudder

    a = self.dive_inner_params.rudder_lpf_alpha
    rudder = a * rudder + (1.0 - a) * self._rudder_lpf_state
    self._rudder_lpf_state = rudder
    return rudder


def apply_aileron_lpf(self, aileron):
    """
    Apply an aileron output LPF to reduce late-dive roll amplification
    as dynamic pressure increases approximately with V^2.

    Apply this on every aileron return path.
    An alpha of 1.0 disables filtering.
    """
    a = self.dive_inner_params.aileron_lpf_alpha
    aileron = a * aileron + (1.0 - a) * self._aileron_lpf_state
    self._aileron_lpf_state = aileron
    return aileron