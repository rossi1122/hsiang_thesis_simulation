"""
RCMQ dive-attack mission, frequency-separated architecture.

Vision Thread (~20 Hz): capture + YOLO in a separate thread
Control Loop  (100 Hz): DKF predict every 10 ms, DKF update on new frames,
                        attitude/rate control never blocked by YOLO
Display       (~20 Hz): every DISPLAY_INTERVAL control ticks
"""

import math
import numpy as np
import time
import cv2
import csv
import random
from collections import deque
from datetime import datetime
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from xplane_connect.xpc3 import XPlaneConnect
from vision_driver import CarDetector, VisionThread, SharedVisionState, VisionResult
from dkf import EgoAwareDKF, reproject_target_ned_to_pixel
from gimbal import GimbalController, RayProjectionDistanceEstimator
from mpc_controller import (
    MPCDiveController,
    DiveMPCParams,
    RCMQ_RUNWAY_18,
    RCMQ_RUNWAY_ELEV,
)
from pid_controller import PIDDiveController, DivePIDParams


class RCMQGimbalDiveAttackAsync:
    """
    RCMQ dive-attack mission.

    Vision Thread: ~20 Hz (YOLO natural rate)
    Control Loop:  100 Hz
    Display:       ~20 Hz
    """

    # ==================== Rates ====================
    CONTROL_DT = 0.01          # control period (100 Hz)
    DISPLAY_INTERVAL = 5       # display every N control ticks (~20 FPS)
    DKF_DT = 0.01              # DKF predict period (same as control)

    # ==================== Outer-loop controller ====================
    # 'MPC' -> mpc_controller.MPCDiveController
    #          3DOF point-mass MPC + terminal visual servo overlay
    # 'PID' -> pid_controller.PIDDiveController
    #          image-error PID, active for the whole DIVE
    # Both expose the same API, so nothing downstream changes.
    # Override per run with the controller_type argument or the
    # DIVE_CONTROLLER environment variable.
    CONTROLLER_TYPE = 'PID'
    #CONTROLLER_TYPE = 'MPC'

    # Parameter overrides applied after the dive-angle derived bounds.
    # Keys must exist on the corresponding params dataclass; a typo raises.
    PID_OVERRIDES = {
    'Kp_lon': 50.0,             
    'pitch_cmd_min_deg': -58.0, 
    }
    MPC_OVERRIDES = {
        # 'w_image_x': 800.0,
        # 'w_image_y': 400.0,
    }

    def __init__(self,
                 model_path,
                 home_heading=179.5,
                 local_start=(24306.17578125, -38588.65625),
                 start_ground_range=-1500,
                 start_elev=201.79 + 300,
                 dive_angle=30.0,
                 enable_dkf=False,
                 image_delay=0.0,
                 target_position=None,
                 vision_buffer_delay_s=0.0,
                 use_dkf_for_control=False,
                 vision_drop_rate=0.0,
                 vision_drop_seed=42,
                 controller_type=None):
        """
        Args:
            dive_angle: dive angle (deg, positive).
            enable_dkf: enable the DKF (recommended True with this architecture).
            image_delay: nominal image delay (s); sizes the DKF history buffer.
            vision_buffer_delay_s: real end-to-end delay injection — YOLO
                results are held in a buffer until age >= this value, so the
                controller really receives N-ms-old positions. 0 = off.
            use_dkf_for_control: False = controller uses raw YOLO when
                detected (DKF is a bystander); True = controller always uses
                the delay-compensated DKF/NED estimate (required for delay
                robustness experiments).
            vision_drop_rate: fraction of detected frames dropped (dropout
                robustness test); fixed seed makes runs reproducible.
        """
        self._client = XPlaneConnect()
        self._home_heading = home_heading
        self._local_start = local_start
        self._start_ground_range = start_ground_range
        self._start_elev = start_elev
        self._dive_angle = dive_angle
        self.image_delay = image_delay

        self.screen_width = 1920
        self.screen_height = 1010

        print("\n" + "="*70)
        print("RCMQ Dive Attack - Async Vision Architecture")
        print("="*70)
        print(f"  Vision Thread:  ~20 Hz (YOLO, separate thread)")
        print(f"  Control Loop:   {1/self.CONTROL_DT:.0f} Hz")
        print(f"  DKF Predict:    {1/self.DKF_DT:.0f} Hz")
        print(f"  Display:        ~{1/self.CONTROL_DT/self.DISPLAY_INTERVAL:.0f} Hz")
        print(f"  dive angle: -{dive_angle}")
        print(f"  DKF: {'on' if enable_dkf else 'off'}")
        print("="*70)

        # Log directory
        self.log_dir = Path(r'C:\Users\hsu\Desktop\hsu_thesis_simulation\Xplane_sim\vision_based_attack\flight_log')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ==================== Components ====================

        # 1. YOLO detector (DKF is managed by the main loop, not the detector)
        self.detector = CarDetector(model_path, use_tracker=False)

        # 2. Shared vision state
        self.shared_vision = SharedVisionState(self.screen_width, self.screen_height)

        # 3. Vision thread
        self.vision_thread = VisionThread(
            self.detector, self.shared_vision,
            self.screen_width, self.screen_height
        )

        # 4. DKF construction deferred until intrinsics exist (below)
        self.enable_dkf = enable_dkf
        self.dkf = None

        # 5. Gimbal controller
        self.gimbal = GimbalController(
            dt=self.CONTROL_DT,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            max_gimbal_pitch=60.0,
            max_gimbal_yaw=45.0,
            gimbal_pitch_rate=30.0,
            gimbal_yaw_rate=45.0,
            camera_offset_x=0.0,
            camera_offset_y=0.0,
            camera_offset_z=-0.0,
            xplane_pitch_calibration_deg=8.5,   # cockpit default-eye tilt bias
        )

        # 6. Range estimator
        self.distance_estimator = RayProjectionDistanceEstimator(
            fov_horizontal=60.0,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            render_width=1920,
            render_height=1080,
            crop_top=30,
            gimbal_pitch_offset_deg=0.0,
            ema_alpha=0.5,   # lower lag for the fast SEARCH approach LoS
        )

        # DKF (needs the estimator intrinsics)
        if enable_dkf:
            # Compensation depth follows the injected delay (min 30 steps)
            auto_steps = max(30, int((vision_buffer_delay_s + 0.05) / self.DKF_DT))
            self.dkf = EgoAwareDKF(
                dt=self.DKF_DT,
                image_delay=image_delay,
                fx=self.distance_estimator.fx,
                fy=self.distance_estimator.fy,
                cx=self.distance_estimator.cx,
                cy=self.distance_estimator.cy,
                enable_ego_comp=True,
                max_update_delay_steps=auto_steps,
            )
            print(f"[DKF] on (ego comp ON, NED fallback from main loop, "
                  f"compensation cap {auto_steps} steps = {auto_steps*self.DKF_DT*1000:.0f}ms)")
        else:
            self.dkf = None

        # NED fallback trigger parameters
        self.NED_FALLBACK_VISION_AGE_MS = 200.0   # vision older than this = stale
        self.NED_FALLBACK_LOST_FRAMES   = 10      # consecutive DKF misses
        self._ned_fallback_used_count = 0

        # Real delay injection buffer
        self.vision_buffer_delay_s = vision_buffer_delay_s
        self.use_dkf_for_control = use_dkf_for_control

        # Terminal lead: extrapolate the target-KF NED position this many
        # seconds into the future so the aim point compensates the remaining
        # terminal flight time (0 = aim at "now").
        self.terminal_lead_s = 1.4
        self._last_lead_dt_extrap = float('nan')
        self._dkf_display_now = None   # target "now" position (no lead), for display
        # target_kf diagnostics (compare against ground truth post-flight)
        self._dbg_tkf_x = float('nan'); self._dbg_tkf_y = float('nan')
        self._dbg_tkf_vx = float('nan'); self._dbg_tkf_vy = float('nan')
        self._dbg_tkf_mcount = 0

        # Frame-drop simulation. A drop persists until the next real YOLO
        # frame arrives (otherwise held vision results leak through).
        self.vision_drop_rate = max(0.0, min(1.0, vision_drop_rate))
        self._drop_rng = random.Random(vision_drop_seed)
        self._drop_total_eligible = 0
        self._drop_count = 0
        self._currently_dropped = False

        # YOLO rate tracking (raw release rate vs effective post-drop rate)
        self._yolo_raw_arrival_times = deque(maxlen=200)
        self._yolo_effective_arrival_times = deque(maxlen=200)
        self._vision_delay_buffer = deque()
        self._last_released_vision = None   # initialized after screen center

        # DKF output EMA (post-processing only; 1.0 = off)
        self.dkf_output_ema_alpha = 1.0
        self._dkf_smoothed_pos = None

        # Aircraft/gimbal pose history for NED reprojection (100 Hz x 2 s)
        self._pose_history = deque(maxlen=200)
        # NED reprojection main path: historical pose ray-projects delayed
        # YOLO into target_kf NED; current pose reprojects back to pixel.
        self.use_ned_reprojection = True

        # Rate-based slant outlier reject (guards single-frame slant spikes
        # from pushing target_kf away in the late dive)
        self._last_kf_slant_value = None
        self._last_kf_slant_time = None
        self._slant_outlier_reject_count = 0

        if vision_buffer_delay_s > 0:
            print(f"\n{'='*60}")
            print(f"Real delay injection: +{vision_buffer_delay_s*1000:.0f}ms")
            print(f"  YOLO results held in buffer for {vision_buffer_delay_s*1000:.0f}ms")
            print(f"  use_dkf_for_control={use_dkf_for_control} "
                  f"({'NED reproject -> controller' if use_dkf_for_control else 'raw delayed YOLO (control group)'})")
            print(f"{'='*60}\n")

        if self.use_ned_reprojection and use_dkf_for_control:
            print(f"\n{'='*60}")
            print(f"NED reprojection main path enabled")
            print(f"  distance estimator: historical pose at vision timestamp")
            print(f"  controller input: target_kf NED -> current-pose reprojection")
            print(f"  DKF output EMA alpha = {self.dkf_output_ema_alpha} (1.0 = off)")
            print(f"{'='*60}\n")

        if self.vision_drop_rate > 0:
            print(f"\n{'='*60}")
            print(f"Frame-drop simulation: drop_rate = {self.vision_drop_rate*100:.0f}%")
            print(f"  seed = {vision_drop_seed} (reproducible)")
            print(f"{'='*60}\n")

        # 7. Flight controller (MPC or PID outer loop; see CONTROLLER_TYPE).
        self.controller = self._build_controller(dive_angle, controller_type,
                                                 target_position)


        # Main-loop stop altitude, decoupled from RCMQ_RUNWAY_18 (which stays
        # as tch for the DIVE->TERMINAL switch).
        self.stop_altitude_agl = 0.0

        # Manual inner-loop test mode (bypasses MPC + vision). Set True to
        # test pure inner-loop pitch tracking.
        MANUAL_INNER_LOOP_TEST = False
        if MANUAL_INNER_LOOP_TEST:
            self.controller.manual_mode_enabled = True
            self.controller.manual_dive_pitch_deg = -45.0
            self.controller.manual_trigger_alt_agl = None   # None = dive immediately
            print("\n" + "="*60)
            print("MANUAL TEST MODE enabled (MPC + vision bypassed)")
            print(f"  dive pitch cmd: {self.controller.manual_dive_pitch_deg:+.1f}")
            print("="*60 + "\n")

        # Screen center
        self.screen_center_u = self.screen_width / 2
        self.screen_center_v = self.screen_height / 2

        # Latest vision info for control
        self._latest_vision = VisionResult(
            raw_center_x=self.screen_center_u,
            raw_center_y=self.screen_center_v,
        )
        self._vision_is_new = False
        self._last_vision_frame_id = 0

        self._last_released_vision = VisionResult(
            raw_center_x=self.screen_center_u,
            raw_center_y=self.screen_center_v,
        )

        # Data log
        self.data_log = self._init_data_log()

        self.current_bbox = None
        self._dive_start_time = None
        self._last_estimated_distance = None
        self._last_good_lat_offset = 0.0
        self._last_good_lon_offset = 0.0

    def _build_controller(self, dive_angle, controller_type, target_position):
        """
        Build the selected dive controller.

        The pitch bounds are derived from dive_angle for both controllers so
        the two runs are commanded within the same envelope and an MPC-vs-PID
        comparison is apples to apples.

        Priority: controller_type argument > DIVE_CONTROLLER env var >
        CONTROLLER_TYPE class constant.
        """
        kind = controller_type
        if kind is None:
            kind = os.environ.get('DIVE_CONTROLLER', self.CONTROLLER_TYPE)
        kind = str(kind).strip().upper()
        if kind not in ('MPC', 'PID'):
            raise ValueError(f"unknown controller_type {kind!r}; "
                             f"expected 'MPC' or 'PID'")

        dive_target = -abs(dive_angle)
        pitch_min = dive_target - 12.0   # depth margin for image residual
        pitch_max = dive_target + 10.0   # anti-shallow margin

        print(f"\n{'='*60}")
        print(f"Outer-loop controller: {kind}")
        print(f"  dive target {dive_target:+.1f} deg, "
              f"pitch cmd range [{pitch_min:+.1f}, {pitch_max:+.1f}]")

        if kind == 'PID':
            params = DivePIDParams()
            overrides = self.PID_OVERRIDES
        else:
            params = DiveMPCParams()
            overrides = self.MPC_OVERRIDES

        params.dive_pitch_target_deg = dive_target
        params.pitch_cmd_min_deg = pitch_min
        params.pitch_cmd_max_deg = pitch_max

        for key, value in (overrides or {}).items():
            if not hasattr(params, key):
                raise AttributeError(
                    f"{type(params).__name__} has no field {key!r}")
            setattr(params, key, value)
            print(f"  override: {key} = {value}")
        print(f"{'='*60}")

        common = dict(
            gamma=dive_angle,
            dt=self.CONTROL_DT,
            tch=RCMQ_RUNWAY_18,
            runway_elev=RCMQ_RUNWAY_ELEV,
            distance_margin=1.05,
            target_position=target_position,
            ray_estimator=self.distance_estimator,
        )

        if kind == 'PID':
            controller = PIDDiveController(pid_params=params, **common)
        else:
            controller = MPCDiveController(mpc_params=params, **common)

        controller.controller_kind = kind
        self.controller_kind = kind
        return controller


    def _init_data_log(self):
        """Data log keys."""
        keys = [
            'wall_time',
            'time', 'altitude_agl', 'altitude_msl', 'airspeed',
            'pitch', 'pitch_cmd', 'pitch_rate', 'roll', 'heading',
            'yaw_cmd', 'yaw_rate', 'x_position', 'y_position',
            'lateral_offset', 'longitudinal_offset',
            'image_u', 'image_v',
            'rudder', 'elevator', 'aileron', 'throttle',
            'target_detected', 'control_mode',
            'gimbal_pitch', 'gimbal_yaw', 'gimbal_locked',
            'gimbal_target_pitch', 'target_centered',
            'bbox_width', 'bbox_height',
            'vision_fps', 'control_fps',
            'dkf_estimated_u', 'dkf_estimated_v',
            'vision_age_ms',
            # DKF diagnostics
            'raw_yolo_u', 'raw_yolo_v',        # raw YOLO center (NaN if no detection)
            'dkf_std_u', 'dkf_std_v',          # DKF position std (px)
            'is_new_vision',
            # Range diagnostics
            'estimated_distance',
            'distance_confidence',
            'trigger_distance',
            # Run-config metadata (same value every row)
            'dkf_enabled',
            'vision_buffer_delay_ms',
            'use_dkf_for_control',
            'vision_drop_rate',
            'vision_dropped',                  # this tick artificially dropped (0/1)
            'yolo_raw_hz',
            'yolo_effective_hz',
            'ned_fallback_used',
            # TVS diagnostics (calibrate s_enter/s_full from tvs_s_ema;
            # convergence from tvs_u/v_norm; applied correction from tvs_corr_*)
            'tvs_active',
            'tvs_beta',
            'tvs_u_norm',
            'tvs_v_norm',
            'tvs_s_raw',
            'tvs_s_ema',
            'tvs_corr_pitch_deg',
            'tvs_corr_heading_deg',
            'tvs_detected',
            'tvs_edge_flagged',
            'tvs_lost_ticks',
            'tvs_entered',
            # Error-driven beta floor diagnostics
            'tvs_beta_sched',
            'tvs_err_floor',
            'tvs_v_norm_ema',
            # Lead diagnostics
            'lead_dt_extrap',
            'lead_set_s',
            # target_kf NED diagnostics
            'tkf_x', 'tkf_y',
            'tkf_vx', 'tkf_vy',
            'tkf_mcount',
        ]
        return {k: [] for k in keys}

    # ==================== X-Plane interface ====================

    def send_gimbal_to_xplane(self):
        """Send gimbal angles to X-Plane."""
        try:
            head_drefs = self.gimbal.get_xplane_head_drefs()
            self._client.sendDREF("sim/graphics/view/pilots_head_psi", head_drefs['head_psi'])
            self._client.sendDREF("sim/graphics/view/pilots_head_the", head_drefs['head_the'])
            self._client.sendDREF("sim/graphics/view/pilots_head_x", head_drefs['head_x'])
            self._client.sendDREF("sim/graphics/view/pilots_head_y", head_drefs['head_y'])
            self._client.sendDREF("sim/graphics/view/pilots_head_z", head_drefs['head_z'])
        except Exception as e:
            print(f"[Gimbal] X-Plane comm error: {e}")

    def reset(self, cteInit=0, heInit=0, dtpInit=0, noBrake=True):
        """Reset the aircraft to the initial position."""
        print("\nResetting aircraft...")

        if self.vision_thread.is_running:
            self.vision_thread.stop()

        self._client.pauseSim(True)
        self._client.sendDREF("sim/operation/override/override_joystick", 1)
        self._client.sendCTRL([0, 0, 0, 0])
        self._client.sendDREF("sim/flightmodel/controls/parkbrake", int(noBrake))

        # Zero dynamics
        initRef = "sim/flightmodel/position/"
        drefs = []
        refs = ['theta', 'phi', 'local_vx', 'local_vy', 'local_vz',
                'local_ax', 'local_ay', 'local_az',
                'Prad', 'Qrad', 'Rrad', 'q', 'groundspeed',
                'indicated_airspeed', 'indicated_airspeed2', 'true_airspeed',
                'M', 'N', 'L', 'P', 'Q', 'R', 'P_dot', 'Q_dot', 'R_dot',
                'Prad', 'Qrad', 'Rrad']
        for ref in refs:
            drefs += [initRef + ref]
        values = [0] * len(refs)
        self._client.sendDREFs(drefs, values)

        init_pitch = 3.0
        self.set_orient_pos(0., init_pitch, 0, self._start_ground_range, 0, self._start_elev)

        self._client.sendDREFs(["sim/operation/fix_all_systems"], [1])
        self._client.sendDREF("sim/flightmodel/engine/ENGN_mixt", 0.61)

        initial_speed = 30
        heading = self._home_heading - heInit
        pitch_rad = math.radians(init_pitch)
        horiz_speed = initial_speed * math.cos(pitch_rad)
        vert_speed = initial_speed * math.sin(pitch_rad)
        self._client.sendDREF("sim/flightmodel/position/local_vx",
                              horiz_speed * np.sin(heading * np.pi / 180.0))
        self._client.sendDREF("sim/flightmodel/position/local_vz",
                              -horiz_speed * np.cos(heading * np.pi / 180.0))
        self._client.sendDREF("sim/flightmodel/position/local_vy", vert_speed)

        self._client.sendCTRL([0.05, 0, 0, 0.5])

        self._client.sendDREFs(
            ["sim/flightmodel/weight/m_fuel1", "sim/flightmodel/weight/m_fuel2"],
            [232, 232]
        )

        # Reset components
        self.controller.reset_controllers()
        self.gimbal.reset(reset_position=False)
        self.distance_estimator.reset()

        if self.dkf is not None:
            self.dkf.reset()
        self._ned_fallback_used_count = 0
        self._prev_dkf_mode = None
        self._dkf_smoothed_pos = None

        # Reset gimbal view
        self._client.sendDREF("sim/graphics/view/pilots_head_psi", 0.0)
        self._client.sendDREF("sim/graphics/view/pilots_head_the", 0.0)
        self._client.sendDREF("sim/graphics/view/pilots_head_x", self.gimbal.camera_offset_x)
        self._client.sendDREF("sim/graphics/view/pilots_head_y", self.gimbal.camera_offset_y)
        self._client.sendDREF("sim/graphics/view/pilots_head_z", self.gimbal.camera_offset_z)

        self.data_log = self._init_data_log()

        self._latest_vision = VisionResult(
            raw_center_x=self.screen_center_u,
            raw_center_y=self.screen_center_v,
        )
        self._vision_is_new = False
        self._last_vision_frame_id = 0
        self.current_bbox = None
        self._dive_start_time = None
        self._last_good_lat_offset = 0.0
        self._last_good_lon_offset = 0.0

        time.sleep(2)
        print("Aircraft reset done")

    # ==================== Coordinate transforms ====================

    def set_orient_pos(self, phi, theta, psi, x, y, h):
        self._client.sendDREF('sim/flightmodel/position/phi', phi)
        self._client.sendDREF('sim/flightmodel/position/theta', theta)
        self._client.sendDREF('sim/flightmodel/position/psi', self.To_local_heading(psi))
        self.setHomeState(x, y)
        curr_elev = self._client.getDREF("sim/flightmodel/position/elevation")[0]
        curr_localy = self._client.getDREF("sim/flightmodel/position/local_y")[0]
        offset = curr_elev - curr_localy
        self._client.sendDREF("sim/flightmodel/position/local_y", h - offset)

    def To_local_heading(self, psi):
        return psi + self._home_heading

    def setHomeState(self, x, y):
        y_new = -y
        localz, localx = self.homeToLocal(x, y_new)
        self._client.sendDREF("sim/flightmodel/position/local_x", localx)
        self._client.sendDREF("sim/flightmodel/position/local_z", localz)

    def homeToLocal(self, x, y):
        roty, rotx = self.rotateToLocal(y, x)
        start_X, start_Y = self._local_start
        return start_X + rotx, start_Y + roty

    def rotateToLocal(self, y, x):
        a = math.cos(math.radians(-3.327831058162679))
        b = math.sin(math.radians(-3.327831058162679))
        return a * y - b * x, b * y + a * x

    def get_home_xy(self):
        x = self._client.getDREF("sim/flightmodel/position/local_z")[0]
        y = self._client.getDREF("sim/flightmodel/position/local_x")[0]
        rotx, roty = self.LocalTohome(x, y)
        return rotx, -roty

    def LocalTohome(self, x, y):
        start_X, start_Y = self._local_start
        return self.rotate_to_home(x - start_X, y - start_Y)

    def rotate_to_home(self, y, x):
        a = math.cos(math.radians(-3.327831058162679))
        b = math.sin(math.radians(-3.327831058162679))
        return a * y - b * x, b * y + a * x

    def get_statevec(self):
        airspeed = self._client.getDREF("sim/flightmodel/position/true_airspeed")[0]
        airspeed_knot = airspeed * 1.9438

        elevator_surface = self._client.getDREF('sim/flightmodel/controls/hstab1_elv1def')[0]
        aileron_surface = self._client.getDREF('sim/flightmodel/controls/wing1l_ail1def')[0]
        rudder_surface = self._client.getDREF('sim/flightmodel/controls/vstab1_rud1def')[0]

        vel = self.body_frame_velocity()
        P = self._client.getDREF('sim/flightmodel/position/P')[0]
        Q = self._client.getDREF('sim/flightmodel/position/Q')[0]
        R = self._client.getDREF('sim/flightmodel/position/R')[0]

        phi = self._client.getDREF('sim/flightmodel/position/phi')[0]
        theta = self._client.getDREF('sim/flightmodel/position/theta')[0]
        psi = self.get_home_heading()

        alpha = self._client.getDREF('sim/flightmodel/position/alpha')[0]
        beta = self._client.getDREF('sim/flightmodel/position/beta')[0]
        gamma_true = self._client.getDREF('sim/flightmodel/position/vpath')[0]
        gamma = theta - alpha
        vision_gamma = 5
        x, y = self.get_home_xy()
        h = self._client.getDREF('sim/flightmodel/position/elevation')[0]

        return np.array([
            airspeed_knot, elevator_surface, aileron_surface, rudder_surface,
            vel[0], vel[1], vel[2], P, Q, R, phi, theta, psi, alpha, beta,
            gamma_true, vision_gamma, x, y, h
        ]).T

    def body_frame_velocity(self):
        cos = math.cos
        sin = math.sin

        psi = self._client.getDREF('sim/flightmodel/position/psi')[0]
        theta = self._client.getDREF('sim/flightmodel/position/theta')[0]
        phi = self._client.getDREF('sim/flightmodel/position/phi')[0]

        h = math.radians(psi)
        Rh = np.array([[cos(h), sin(h), 0], [-sin(h), cos(h), 0], [0, 0, 1]])
        el = math.radians(theta)
        Re = np.array([[cos(el), 0, -sin(el)], [0, 1, 0], [sin(el), 0, cos(el)]])
        roll = math.radians(phi)
        Rr = np.array([[1, 0, 0], [0, cos(roll), sin(roll)], [0, -sin(roll), cos(roll)]])
        R = np.matmul(Rr, np.matmul(Re, Rh))

        vx = self._client.getDREF('sim/flightmodel/position/local_vx')[0]
        vy = self._client.getDREF('sim/flightmodel/position/local_vy')[0]
        vz = self._client.getDREF('sim/flightmodel/position/local_vz')[0]

        return np.matmul(R, np.array([-vz, vx, -vy]).T)

    def get_ned_state(self):
        """
        Aircraft NED position/velocity/heading.

        X-Plane OpenGL: local_x=East, local_y=Up, local_z=South (-North):
            x_ned (N) = -local_z, y_ned (E) = +local_x
        """
        local_x = self._client.getDREF('sim/flightmodel/position/local_x')[0]
        local_z = self._client.getDREF('sim/flightmodel/position/local_z')[0]
        local_vx = self._client.getDREF('sim/flightmodel/position/local_vx')[0]
        local_vz = self._client.getDREF('sim/flightmodel/position/local_vz')[0]
        true_psi_deg = self._client.getDREF('sim/flightmodel/position/psi')[0]
        return {
            'x_ned':   -local_z,
            'y_ned':    local_x,
            'vx_ned':  -local_vz,
            'vy_ned':   local_vx,
            'heading_rad': math.radians(true_psi_deg),
        }

    def _calc_yolo_rate(self, dq, window_s=1.0):
        """Frame rate (Hz) over the last window_s seconds."""
        if not dq:
            return 0.0
        now = time.time()
        while dq and now - dq[0] > window_s:
            dq.popleft()
        return len(dq) / window_s

    def get_home_heading(self):
        true_heading = self._client.getDREF("sim/flightmodel/position/psi")[0]
        return true_heading - self._home_heading

    def _find_pose_at_time(self, target_t):
        """
        Pose snapshot closest to target_t. Delayed YOLO must be ray-projected
        with the pose at capture time, not the current pose.
        """
        if len(self._pose_history) == 0:
            return None
        return min(self._pose_history, key=lambda p: abs(p['t'] - target_t))

    def _snapshot_pose(self, ned_state, altitude_agl, phi, theta, true_heading_deg):
        """Push the current aircraft+gimbal pose into the history."""
        self._pose_history.append({
            't':                 time.time(),
            'x_ned':             ned_state['x_ned'],
            'y_ned':             ned_state['y_ned'],
            'vx_ned':            ned_state['vx_ned'],
            'vy_ned':            ned_state['vy_ned'],
            'altitude_agl':      altitude_agl,
            'roll_deg':          phi,
            'pitch_deg':         theta,
            'yaw_true_deg':      true_heading_deg,
            'gimbal_pitch_deg':  self.gimbal.gimbal_pitch,
            'gimbal_yaw_deg':    self.gimbal.gimbal_yaw,
        })

    def pause(self, yes=True):
        self._client.pauseSim(yes)

    def send_ctrl(self, elev, aileron, rudder, throttle):
        self._client.sendCTRL([elev, aileron, rudder, throttle])

    # ==================== Delay injection ====================

    def _get_delayed_vision(self):
        """
        Latest YOLO through the delay buffer.

        New results are enqueued; only results with age >= the delay are
        released; with no new release the last released result is returned.
        delay = 0 passes shared_vision.get_latest() straight through.

        Returns:
            (delayed_result, is_new). The result keeps its original capture
            timestamp, so vision_age = buffer delay + YOLO inference delay.
        """
        real_result, real_is_new = self.shared_vision.get_latest()
        if real_is_new and self.vision_buffer_delay_s > 0:
            self._vision_delay_buffer.append(real_result)

        if self.vision_buffer_delay_s <= 0:
            return real_result, real_is_new

        now = time.time()
        just_released = False
        while self._vision_delay_buffer:
            head = self._vision_delay_buffer[0]
            if (now - head.timestamp) >= self.vision_buffer_delay_s:
                self._last_released_vision = self._vision_delay_buffer.popleft()
                just_released = True
            else:
                break

        return self._last_released_vision, just_released

    # ==================== DKF processing ====================

    def process_dkf(self,
                    vision_result: VisionResult,
                    is_new_vision: bool,
                    ned_state=None,
                    altitude_agl=None,
                    ac_roll_deg=None,
                    ac_pitch_deg=None,
                    gimbal_pitch_deg=None,
                    gimbal_yaw_deg=None,
                    slant_range=None):
        """
        High-rate DKF step: projective ego comp + delayed update +
        NED fallback + NED reprojection main path.

        Returns the pixel estimate fed to the controller.
        """
        if self.dkf is None:
            if vision_result.detected:
                return vision_result.raw_center_x, vision_result.raw_center_y
            return self.screen_center_u, self.screen_center_v

        effective_timestamp = vision_result.timestamp

        # PRE_DIVE -> DIVE: reset DKF velocity. PRE_DIVE velocity is the
        # ego-compensated residual (small); after DIVE ego comp is off and
        # velocity must rebuild (pure CV includes aircraft motion).
        current_mode = getattr(self.controller, 'current_mode', None)
        if not hasattr(self, '_prev_dkf_mode'):
            self._prev_dkf_mode = current_mode
        if (self._prev_dkf_mode == 'PRE_DIVE' and current_mode == 'DIVE'
                and self.dkf.initialized):
            self.dkf.x[2:4] = 0.0
            self.dkf.P[2, 2] = 50.0**2
            self.dkf.P[3, 3] = 50.0**2
            self.dkf.P[0, 2] = 0.0; self.dkf.P[2, 0] = 0.0
            self.dkf.P[1, 3] = 0.0; self.dkf.P[3, 1] = 0.0
            print(f"[DKF] PRE_DIVE -> DIVE: velocity reset, P inflated")
        self._prev_dkf_mode = current_mode

        # ── Step 1: predict ──
        # DIVE disables ego comp: gimbal-lock reference jumps + unstable
        # close-range depth systematically over-correct the V axis; pure CV
        # with direct YOLO correction measured better in DIVE.
        disable_ego_for_dive = (current_mode == 'DIVE')

        if (ned_state is not None and altitude_agl is not None
                and not disable_ego_for_dive):
            ac_yaw_true_deg = math.degrees(ned_state['heading_rad'])
            ac_state = dict(
                x_ned=ned_state['x_ned'],
                y_ned=ned_state['y_ned'],
                alt_agl=altitude_agl,
                roll_deg=ac_roll_deg,
                pitch_deg=ac_pitch_deg,
                yaw_true_deg=ac_yaw_true_deg,
            )
            gimbal_state = dict(
                pitch_deg=gimbal_pitch_deg,
                yaw_deg=gimbal_yaw_deg,
            )
            depth = slant_range if (slant_range and slant_range > 1.0) else None
            self.dkf.predict(
                ac_state=ac_state,
                gimbal_state=gimbal_state,
                depth_m=depth,
            )
        else:
            self.dkf.predict()
            # Clear prev state so a later ego-comp restart has no stale reference
            if disable_ego_for_dive:
                self.dkf._prev_ac_state = None
                self.dkf._prev_gimbal_state = None

        # ── Step 2: YOLO update (with delay compensation) ──
        if is_new_vision and vision_result.detected:
            z_image = np.array([
                float(vision_result.raw_center_x),
                float(vision_result.raw_center_y)
            ])
            if effective_timestamp > 0:
                actual_delay_sec = time.time() - effective_timestamp
                delay_steps = max(1, int(actual_delay_sec / self.DKF_DT))
            else:
                delay_steps = None
            bbox_diag = math.sqrt(vision_result.bbox_width**2 + vision_result.bbox_height**2) \
                        if vision_result.bbox_width > 0 else None
            self.dkf.update(z_image, delay_steps=delay_steps, bbox_size=bbox_diag)

        # ── Step 3: NED back-projection fallback (severe loss) ──
        if ned_state is not None and altitude_agl is not None:
            vision_age_ms = (time.time() - effective_timestamp) * 1000 \
                            if effective_timestamp > 0 else 1e9
            target_kf = self.controller.target_kf
            target_lost = (not vision_result.detected)
            vision_stale = vision_age_ms > self.NED_FALLBACK_VISION_AGE_MS
            severe = self.dkf._consecutive_miss >= self.NED_FALLBACK_LOST_FRAMES

            if (target_lost or vision_stale) and severe and target_kf._initialized:
                de = self.distance_estimator
                u_proj, v_proj, valid = reproject_target_ned_to_pixel(
                    target_x_ned=float(target_kf.x[0]),
                    target_y_ned=float(target_kf.x[1]),
                    ac_x_ned=ned_state['x_ned'],
                    ac_y_ned=ned_state['y_ned'],
                    ac_altitude_agl=altitude_agl,
                    ac_roll_deg=ac_roll_deg,
                    ac_pitch_deg=ac_pitch_deg,
                    ac_yaw_true_deg=math.degrees(ned_state['heading_rad']),
                    gimbal_pitch_deg=gimbal_pitch_deg,
                    gimbal_yaw_deg=gimbal_yaw_deg,
                    fx=de.fx, fy=de.fy, cx=de.cx, cy=de.cy,
                    R_g2c=de.R_g2c,
                    gimbal_pitch_offset_deg=de.gimbal_pitch_offset_deg,
                )

                if valid:
                    # R estimate: reprojection error ~ (fx / depth) * pos std
                    pos_var_m2 = float(target_kf.P[0, 0] + target_kf.P[1, 1])
                    pos_std_m = math.sqrt(max(pos_var_m2, 1.0))
                    depth_for_r = max(slant_range or 100.0, 50.0)
                    r_std_px = (de.fx / depth_for_r) * pos_std_m + 5.0
                    r_std_px = float(np.clip(r_std_px, 5.0, 100.0))

                    self.dkf.update_fallback(
                        np.array([u_proj, v_proj]),
                        r_std_pixel=r_std_px,
                    )
                    self._ned_fallback_used_count += 1
                    if self._ned_fallback_used_count % 50 == 1:
                        print(f"[DKF] NED fallback "
                              f"(vision_age={vision_age_ms:.0f}ms, "
                              f"miss={self.dkf._consecutive_miss}, "
                              f"r_std={r_std_px:.1f}px, count={self._ned_fallback_used_count})")

        # ── NED reprojection main path ──
        # distance estimator feeds target_kf NED via historical pose;
        # here the current target_kf NED state is reprojected with the
        # CURRENT pose = "where the target should be now" for the controller.
        # Pure geometry; no pixel-KF extrapolation, no ego-shift model.
        target_kf = (self.controller.target_kf
                     if (self.controller is not None
                         and hasattr(self.controller, 'target_kf'))
                     else None)
        ned_path_used = False

        if (self.use_dkf_for_control
                and self.use_ned_reprojection
                and target_kf is not None
                and target_kf._initialized
                and len(self._pose_history) > 0):

            cur_pose = self._pose_history[-1]
            de = self.distance_estimator

            # target_kf.x is the estimate at the (delayed) measurement time;
            # propagate along velocity to now, plus terminal lead.
            now = time.time()
            last_meas_t = target_kf._last_update_time
            dt_extrap = max(0.0, now - last_meas_t)
            dt_extrap = dt_extrap + self.terminal_lead_s
            # Cap guards unconverged KF velocity; must include the lead
            dt_extrap = min(dt_extrap, self.vision_buffer_delay_s + self.terminal_lead_s + 0.2)
            self._last_lead_dt_extrap = dt_extrap

            # Velocity unreliable below 3 measurements -> no propagation
            if target_kf._measurement_count >= 3:
                tgt_x_now = float(target_kf.x[0]) + float(target_kf.x[2]) * dt_extrap
                tgt_y_now = float(target_kf.x[1]) + float(target_kf.x[3]) * dt_extrap
            else:
                tgt_x_now = float(target_kf.x[0])
                tgt_y_now = float(target_kf.x[1])

            # Diagnostics for post-flight comparison against ground truth
            self._dbg_tkf_x  = float(target_kf.x[0])
            self._dbg_tkf_y  = float(target_kf.x[1])
            self._dbg_tkf_vx = float(target_kf.x[2])
            self._dbg_tkf_vy = float(target_kf.x[3])
            self._dbg_tkf_mcount = int(target_kf._measurement_count)

            u_proj, v_proj, valid = reproject_target_ned_to_pixel(
                target_x_ned=tgt_x_now,
                target_y_ned=tgt_y_now,
                ac_x_ned=cur_pose['x_ned'],
                ac_y_ned=cur_pose['y_ned'],
                ac_altitude_agl=cur_pose['altitude_agl'],
                ac_roll_deg=cur_pose['roll_deg'],
                ac_pitch_deg=cur_pose['pitch_deg'],
                ac_yaw_true_deg=cur_pose['yaw_true_deg'],
                gimbal_pitch_deg=cur_pose['gimbal_pitch_deg'],
                gimbal_yaw_deg=cur_pose['gimbal_yaw_deg'],
                fx=de.fx, fy=de.fy, cx=de.cx, cy=de.cy,
                R_g2c=de.R_g2c,
                gimbal_pitch_offset_deg=de.gimbal_pitch_offset_deg,
            )
            if valid:
                u_raw = u_proj
                v_raw = v_proj
                ned_path_used = True

            # Display-only: target "now" WITHOUT the lead (the lead point
            # runs ahead of the vehicle; the green box should sit on it).
            dt_now = min(max(0.0, now - last_meas_t),
                         self.vision_buffer_delay_s + 0.2)
            if target_kf._measurement_count >= 3:
                tgt_x_disp = float(target_kf.x[0]) + float(target_kf.x[2]) * dt_now
                tgt_y_disp = float(target_kf.x[1]) + float(target_kf.x[3]) * dt_now
            else:
                tgt_x_disp = float(target_kf.x[0])
                tgt_y_disp = float(target_kf.x[1])
            u_disp, v_disp, valid_disp = reproject_target_ned_to_pixel(
                target_x_ned=tgt_x_disp,
                target_y_ned=tgt_y_disp,
                ac_x_ned=cur_pose['x_ned'],
                ac_y_ned=cur_pose['y_ned'],
                ac_altitude_agl=cur_pose['altitude_agl'],
                ac_roll_deg=cur_pose['roll_deg'],
                ac_pitch_deg=cur_pose['pitch_deg'],
                ac_yaw_true_deg=cur_pose['yaw_true_deg'],
                gimbal_pitch_deg=cur_pose['gimbal_pitch_deg'],
                gimbal_yaw_deg=cur_pose['gimbal_yaw_deg'],
                fx=de.fx, fy=de.fy, cx=de.cx, cy=de.cy,
                R_g2c=de.R_g2c,
                gimbal_pitch_offset_deg=de.gimbal_pitch_offset_deg,
            )
            self._dkf_display_now = (u_disp, v_disp) if valid_disp else None

        if not ned_path_used:
            # Fallbacks: target_kf not initialized / use_dkf_for_control off
            # (control group) / reprojection invalid (target behind camera)
            if vision_result.detected:
                u_raw = float(vision_result.raw_center_x)
                v_raw = float(vision_result.raw_center_y)
            else:
                pos = self.dkf.get_position()
                u_raw, v_raw = float(pos[0]), float(pos[1])

        # Output EMA (post-processing only; first call initializes directly)
        if self._dkf_smoothed_pos is None:
            self._dkf_smoothed_pos = (u_raw, v_raw)
        else:
            a = self.dkf_output_ema_alpha
            us, vs = self._dkf_smoothed_pos
            self._dkf_smoothed_pos = (a*u_raw + (1-a)*us, a*v_raw + (1-a)*vs)

        return int(self._dkf_smoothed_pos[0]), int(self._dkf_smoothed_pos[1])

    # ==================== Offset ====================

    def calculate_offset(self, target_x, target_y):
        """Half-screen normalized offsets in [-1, 1]."""
        pixel_offset_x = target_x - self.screen_center_u
        pixel_offset_y = target_y - self.screen_center_v

        normalized_x = np.clip(pixel_offset_x / (self.screen_width / 2), -1.0, 1.0)
        normalized_y = np.clip(pixel_offset_y / (self.screen_height / 2), -1.0, 1.0)

        return normalized_x, normalized_y

    # ==================== Main mission loop ====================

    def run_mission(self, max_time=180.0, show_debug=True):
        """
        Run the mission. Vision thread ~20 Hz; this loop 100 Hz.
        """
        print("\n" + "="*70)
        print("Mission start")
        print("="*70)
        print("Keys: q=quit  r=reset  g=lock/unlock gimbal")
        print("      camera: w/s/a/d/z/x move, 0 reset")
        print("="*70 + "\n")

        self.reset()

        dt = self.CONTROL_DT
        current_time = 0.0
        control_step = 0

        ctrl_fps = 0
        ctrl_fps_count = 0
        ctrl_fps_timer = time.time()

        try:
            self.vision_thread.start()
            time.sleep(0.5)   # wait for the first frame

            self.pause(False)

            while current_time < max_time:
                loop_start = time.time()

                # ========== 1. Aircraft state ==========
                state = self.get_statevec()
                h = state[19]
                altitude_agl = h - RCMQ_RUNWAY_ELEV
                x, y = self.get_home_xy()
                heading = state[12]
                Q_rate = state[8]
                R_rate = state[9]
                P_rate = state[7]
                theta = state[11]
                phi = state[10]

                self.detector.update_altitude(altitude_agl)

                if altitude_agl < self.stop_altitude_agl:
                    print(f"\nStop altitude reached ({self.stop_altitude_agl:.1f}m AGL)")
                    break

                # Pose snapshot every tick (NED reprojection needs history)
                _ned_now = self.get_ned_state()
                _true_heading_now = heading + self._home_heading
                self._snapshot_pose(_ned_now, altitude_agl, phi, theta, _true_heading_now)

                # ========== 2. Vision (non-blocking, via delay buffer) ==========
                vision_result, is_new_vision = self._get_delayed_vision()

                # Raw YOLO release rate (captured before the drop mask)
                _now_ts = time.time()
                if is_new_vision:
                    self._yolo_raw_arrival_times.append(_now_ts)

                # Frame-drop simulation: the arrival of a NEW frame decides
                # drop/keep (Bernoulli); ticks in between keep the last state,
                # so a drop persists across the whole YOLO interval.
                if self.vision_drop_rate > 0:
                    if is_new_vision and vision_result.detected:
                        self._drop_total_eligible += 1
                        if self._drop_rng.random() < self.vision_drop_rate:
                            self._currently_dropped = True
                            self._drop_count += 1
                        else:
                            self._currently_dropped = False
                    elif is_new_vision and not vision_result.detected:
                        # YOLO's own miss -> also treated as dropped
                        self._currently_dropped = True

                _vision_dropped_this_tick = (self.vision_drop_rate > 0
                                              and self._currently_dropped)
                if _vision_dropped_this_tick:
                    vision_result = VisionResult(
                        timestamp=vision_result.timestamp,
                        detected=False,
                        raw_center_x=self.screen_center_u,
                        raw_center_y=self.screen_center_v,
                    )
                    is_new_vision = False

                # Effective rate (post-drop, detected=True)
                if is_new_vision and vision_result.detected:
                    self._yolo_effective_arrival_times.append(_now_ts)

                if is_new_vision and vision_result.bbox is not None:
                    self.current_bbox = vision_result.bbox

                detected = vision_result.detected
                raw_center_x = vision_result.raw_center_x
                raw_center_y = vision_result.raw_center_y
                bbox_width = vision_result.bbox_width
                bbox_height = vision_result.bbox_height

                vision_age_ms = (time.time() - vision_result.timestamp) * 1000 if vision_result.timestamp > 0 else 0

                # ========== 3. DKF (every control tick) ==========
                estimated_x, estimated_y = self.process_dkf(
                    vision_result, is_new_vision,
                    ned_state=_ned_now,
                    altitude_agl=altitude_agl,
                    ac_roll_deg=phi,
                    ac_pitch_deg=theta,
                    gimbal_pitch_deg=self.gimbal.gimbal_pitch,   # previous-tick angles
                    gimbal_yaw_deg=self.gimbal.gimbal_yaw,
                    slant_range=self._last_estimated_distance,   # previous-tick range
                )

                # use_dkf_for_control=False: raw YOLO when detected (no filter delay)
                # use_dkf_for_control=True: always the delay-compensated estimate
                current_mode = self.controller.current_mode
                if detected and not self.use_dkf_for_control:
                    estimated_x = int(raw_center_x)
                    estimated_y = int(raw_center_y)

                # DKF drift -> MPC, non-DIVE only (DIVE velocity may be wrong)
                if self.dkf is not None and self.dkf.initialized and current_mode != 'DIVE':
                    dkf_vel = self.dkf.get_velocity()
                    self.controller.update_dkf_drift(
                        u_dot_px=dkf_vel[0],
                        v_dot_px=dkf_vel[1],
                        pitch_rate_deg=Q_rate,
                        heading_rate_deg=R_rate,
                    )

                # Offsets
                if detected:
                    lateral_offset, longitudinal_offset = self.calculate_offset(estimated_x, estimated_y)
                    self._last_good_lat_offset = lateral_offset
                    self._last_good_lon_offset = longitudinal_offset
                elif current_mode == 'DIVE':
                    # DIVE + lost -> hold last good offsets
                    lateral_offset = self._last_good_lat_offset
                    longitudinal_offset = self._last_good_lon_offset
                elif self.dkf is not None and self.dkf.initialized:
                    lateral_offset, longitudinal_offset = self.calculate_offset(estimated_x, estimated_y)
                else:
                    lateral_offset, longitudinal_offset = 0.0, 0.0

                # ========== 4. Gimbal ==========
                current_mode = self.controller.current_mode
                gimbal_target_pitch = 0.0

                self.gimbal.set_predive_mode(current_mode == 'PRE_DIVE')

                if current_mode != 'DIVE':
                    self._dive_start_time = None

                if current_mode == 'SEARCH':
                    # Track only on new frames (repeated corrections on stale
                    # data oscillate). Gimbal tracks the NED-reprojected
                    # estimate, not raw YOLO, so the view stays on the target
                    # under delay.
                    if is_new_vision:
                        gimbal_pitch, gimbal_yaw, target_centered = \
                            self.gimbal.track_target(
                                estimated_x, estimated_y, detected,
                                aircraft_q=Q_rate,
                                aircraft_p=P_rate,
                            )
                    else:
                        gimbal_pitch = self.gimbal.gimbal_pitch
                        gimbal_yaw = self.gimbal.gimbal_yaw
                        target_centered = False

                elif current_mode == 'PRE_DIVE':
                    # Layered stabilization: time-based LoS transition +
                    # 3-axis compensation + pixel fine-tracking.
                    # vision_age_ms=0: estimated is already lag-compensated,
                    # avoid double compensation inside stabilize.
                    true_heading_for_stab = heading + self._home_heading
                    gimbal_pitch, gimbal_yaw, target_centered = \
                        self.gimbal.stabilize_with_tracking(
                            aircraft_pitch=theta,
                            aircraft_roll=phi,
                            aircraft_heading=true_heading_for_stab,
                            target_x=estimated_x,
                            target_y=estimated_y,
                            target_detected=detected,
                            is_new_vision=is_new_vision,
                            dive_angle=self.controller.get_current_dive_pitch_base(),
                            current_time=current_time,
                            vision_age_ms=0.0,
                        )

                    # Late PRE_DIVE: slew gimbal yaw to zero (pitch is already
                    # clipped at 0), so the DIVE lock(0,0) causes no image jump
                    _target_pitch = self.controller.pre_dive_pitch_target
                    if theta <= (_target_pitch + 5.0):
                        if hasattr(self.gimbal, '_stab_yaw_target'):
                            self.gimbal._stab_yaw_target = 0.0
                        _yaw_slew = 20.0 * 0.01   # 20 deg/s x dt
                        if abs(self.gimbal.gimbal_yaw) > 0.05:
                            _step = max(-_yaw_slew, min(_yaw_slew, -self.gimbal.gimbal_yaw))
                            self.gimbal.gimbal_yaw += _step
                        else:
                            self.gimbal.gimbal_yaw = 0.0
                        gimbal_yaw = self.gimbal.gimbal_yaw

                    # DIVE pre-lock: PRE_DIVE completed this tick -> init DIVE
                    # now so the controller's first dive_control tick never
                    # runs with an unlocked gimbal / unreset cache
                    if self.controller.pre_dive_complete and self._dive_start_time is None:
                        self._dive_start_time = current_time
                        self.controller.notify_dive_started(timestamp=time.time())
                        # Lock at the current angles (no snap, no image_v jump)
                        self.gimbal.lock_gimbal(at_zero=True)
                        _locked_pitch = self.gimbal.gimbal_pitch
                        _locked_yaw   = self.gimbal.gimbal_yaw
                        self.controller.notify_gimbal_locked(
                            timestamp=time.time(),
                            gimbal_pitch=_locked_pitch,
                            gimbal_yaw=_locked_yaw,
                        )
                        print(f"[Gimbal] PRE_DIVE->DIVE pre-lock at "
                              f"({_locked_pitch:+.2f}, {_locked_yaw:+.2f}) - no snap")
                        gimbal_pitch = _locked_pitch
                        gimbal_yaw   = _locked_yaw
                        target_centered = True
                        current_mode = 'DIVE'   # display/log follow

                    gimbal_target_pitch = gimbal_pitch

                elif current_mode == 'DIVE':
                    # Lock gimbal on DIVE entry (no snap)
                    if self._dive_start_time is None:
                        self._dive_start_time = current_time
                        self.controller.notify_dive_started(timestamp=time.time())
                        self.gimbal.lock_gimbal(at_zero=True)
                        _locked_pitch = self.gimbal.gimbal_pitch
                        _locked_yaw   = self.gimbal.gimbal_yaw
                        self.controller.notify_gimbal_locked(
                            timestamp=time.time(),
                            gimbal_pitch=_locked_pitch,
                            gimbal_yaw=_locked_yaw,
                        )
                        print(f"[Gimbal] DIVE lock at "
                              f"({_locked_pitch:+.2f}, {_locked_yaw:+.2f}) - no snap")

                    gimbal_pitch = self.gimbal.gimbal_pitch
                    gimbal_yaw = self.gimbal.gimbal_yaw
                    target_centered = True
                else:
                    gimbal_pitch = self.gimbal.gimbal_pitch
                    gimbal_yaw = self.gimbal.gimbal_yaw
                    target_centered = False

                self.send_gimbal_to_xplane()

                # ========== 5. Range estimation (new frames only) ==========
                # Use the historical pose at vision_result.timestamp; the
                # current pose paired with a delayed pixel gives a wrong NED.
                estimated_distance = None
                distance_confidence = 0.0

                if detected and is_new_vision:
                    _vision_t = vision_result.timestamp
                    _hist_pose = self._find_pose_at_time(_vision_t) if _vision_t > 0 else None

                    # Verify the found pose is actually close to vision_t.
                    # Early SEARCH the history hasn't covered the delay yet
                    # and _find_pose_at_time falls back to the oldest entry.
                    # Tolerance: 10 ms sampling x5 = 50 ms.
                    _pose_age_ok = True
                    if _hist_pose is not None and _vision_t > 0:
                        _pose_t_diff = abs(_hist_pose['t'] - _vision_t)
                        if _pose_t_diff > 0.05:
                            _pose_age_ok = False
                            if not hasattr(self, '_pose_match_skip_count'):
                                self._pose_match_skip_count = 0
                            self._pose_match_skip_count += 1
                            if self._pose_match_skip_count % 20 == 1:
                                print(f"[Pose] history does not cover vision_t "
                                      f"(off by {_pose_t_diff*1000:.0f}ms), skip KF update "
                                      f"(count={self._pose_match_skip_count})")

                    if _hist_pose is None or not _pose_age_ok:
                        # Fall back to current pose for range estimation only;
                        # the KF is NOT fed below.
                        _est_roll  = phi
                        _est_pitch = theta
                        _est_yaw   = heading + self._home_heading
                        _est_gp    = gimbal_pitch
                        _est_gy    = gimbal_yaw
                        _est_alt   = altitude_agl
                        _est_acx   = _ned_now['x_ned']
                        _est_acy   = _ned_now['y_ned']
                        _est_acvx  = _ned_now['vx_ned']
                        _est_acvy  = _ned_now['vy_ned']
                        _est_ac_heading_rad = _ned_now['heading_rad']
                    else:
                        _est_roll  = _hist_pose['roll_deg']
                        _est_pitch = _hist_pose['pitch_deg']
                        _est_yaw   = _hist_pose['yaw_true_deg']
                        _est_gp    = _hist_pose['gimbal_pitch_deg']
                        _est_gy    = _hist_pose['gimbal_yaw_deg']
                        _est_alt   = _hist_pose['altitude_agl']
                        _est_acx   = _hist_pose['x_ned']
                        _est_acy   = _hist_pose['y_ned']
                        _est_acvx  = _hist_pose['vx_ned']
                        _est_acvy  = _hist_pose['vy_ned']
                        _est_ac_heading_rad = math.radians(_hist_pose['yaw_true_deg'])

                    result = self.distance_estimator.estimate_distance(
                        pixel_u=raw_center_x, pixel_v=raw_center_y,
                        aircraft_roll=_est_roll, aircraft_pitch=_est_pitch,
                        aircraft_yaw_true=_est_yaw,
                        gimbal_pitch=_est_gp, gimbal_yaw=_est_gy,
                        altitude_agl=_est_alt
                    )
                    if result[0] is not None:
                        estimated_distance = result[0]
                        distance_confidence = result[2]

                    # Feed the target KF (any mode) when the estimate is valid
                    if (result[0] is not None
                            and distance_confidence > 0.3
                            and self.distance_estimator.last_ray_world is not None
                            and _pose_age_ok):
                        # Use last_raw_slant, not the EMA value: in the dive
                        # the range collapses fast and the EMA lags, pushing
                        # target_kf NED too far. The KF has its own gating.
                        raw_slant = self.distance_estimator.last_raw_slant
                        slant_for_kf = raw_slant if (raw_slant is not None and raw_slant > 0) else result[0]

                        # Rate-based outlier reject: max closure ~80 m/s,
                        # 2.5x margin -> 200 m/s cap kills single-frame spikes
                        slant_ok = True
                        _vt = _vision_t if _vision_t > 0 else time.time()
                        if (self._last_kf_slant_value is not None
                                and self._last_kf_slant_time is not None):
                            dt_slant = max(_vt - self._last_kf_slant_time, 0.01)
                            d_slant = abs(slant_for_kf - self._last_kf_slant_value)
                            slant_rate = d_slant / dt_slant
                            if slant_rate > 200.0:
                                slant_ok = False
                                self._slant_outlier_reject_count += 1
                                if self._slant_outlier_reject_count % 10 == 1:
                                    print(f"[Range] slant outlier reject "
                                          f"({self._last_kf_slant_value:.1f}m -> {slant_for_kf:.1f}m "
                                          f"in {dt_slant*1000:.0f}ms = {slant_rate:.0f}m/s, "
                                          f"count={self._slant_outlier_reject_count})")

                        if slant_ok:
                            self.controller.update_target_world_state(
                                ray_world=self.distance_estimator.last_ray_world,
                                slant_range=slant_for_kf,
                                ground_range=result[1],
                                ac_x_ned=_est_acx,
                                ac_y_ned=_est_acy,
                                ac_vx_ned=_est_acvx,
                                ac_vy_ned=_est_acvy,
                                altitude_agl=_est_alt,
                                aircraft_heading_rad=_est_ac_heading_rad,
                                timestamp=_vt,
                            )
                            self._last_kf_slant_value = slant_for_kf
                            self._last_kf_slant_time = _vt
                elif detected:
                    # No new frame but still tracking -> reuse last estimate
                    estimated_distance = self.distance_estimator._get_current_estimate()
                    distance_confidence = 0.5

                # Current slant override: current target NED (propagated to
                # now) vs current aircraft NED — affects the dive trigger,
                # the display, and the controller depth EMA. Ensures a moving
                # target / fast aircraft never delays the trigger.
                _tkf = (self.controller.target_kf
                        if (self.controller is not None
                            and hasattr(self.controller, 'target_kf'))
                        else None)
                if (self.use_dkf_for_control
                        and self.use_ned_reprojection
                        and _tkf is not None
                        and _tkf._initialized
                        and _tkf._measurement_count >= 3
                        and len(self._pose_history) > 0):
                    _cp = self._pose_history[-1]
                    _now = time.time()
                    _dt_ext = max(0.0, _now - _tkf._last_update_time)
                    _dt_ext = min(_dt_ext, self.vision_buffer_delay_s + 0.2)
                    _tx_now = float(_tkf.x[0]) + float(_tkf.x[2]) * _dt_ext
                    _ty_now = float(_tkf.x[1]) + float(_tkf.x[3]) * _dt_ext
                    _dx = _tx_now - _cp['x_ned']
                    _dy = _ty_now - _cp['y_ned']
                    _horiz = math.sqrt(_dx*_dx + _dy*_dy)
                    _alt   = max(_cp['altitude_agl'], 1.0)
                    _slant_now = math.sqrt(_horiz*_horiz + _alt*_alt)
                    if _slant_now > 1.0 and math.isfinite(_slant_now):
                        estimated_distance = _slant_now
                        distance_confidence = 1.0   # KF-based + propagated

                # Cache for the next DKF tick
                if estimated_distance is not None:
                    self._last_estimated_distance = estimated_distance

                # ========== 6. Flight control (100 Hz) ==========
                # target_valid: DIVE trusts only YOLO or a very short gap;
                # other modes let the DKF bridge longer gaps
                if current_mode == 'DIVE':
                    target_valid = detected or (vision_age_ms < 100)
                else:
                    target_valid = detected or (
                        self.dkf is not None and self.dkf.initialized and vision_age_ms < 500
                    )

                # Feed TVS with the controller-facing center (estimated_x/y:
                # DKF-reprojected when use_dkf_for_control, else raw YOLO).
                # bbox w/h stay raw (proximity schedule only). On dropout TVS
                # holds/decays internally.
                if hasattr(self.controller, 'update_measured_image'):
                    self.controller.update_measured_image(
                        cx=estimated_x, cy=estimated_y,
                        bbox_w=bbox_width, bbox_h=bbox_height,
                        screen_w=self.screen_width, screen_h=self.screen_height,
                        detected=detected,
                        timestamp=vision_result.timestamp,
                        agl=altitude_agl,
                    )

                elevator, aileron, rudder, throttle, pitch_cmd, yaw_cmd = self.controller.control(
                    state, target_valid, lateral_offset, longitudinal_offset,
                    x, current_time,
                    gimbal_pitch=gimbal_pitch, gimbal_yaw=gimbal_yaw,
                    target_centered=target_centered,
                    estimated_distance=estimated_distance,
                    distance_confidence=distance_confidence,
                    y_position=y,
                )

                self.send_ctrl(elevator, aileron, rudder, throttle)

                # ========== 7. Data log ==========
                ctrl_fps_count += 1
                now = time.time()
                if now - ctrl_fps_timer >= 1.0:
                    ctrl_fps = ctrl_fps_count / (now - ctrl_fps_timer)
                    ctrl_fps_count = 0
                    ctrl_fps_timer = now

                self.data_log['wall_time'].append(time.time())
                self.data_log['time'].append(current_time)
                self.data_log['altitude_agl'].append(altitude_agl)
                self.data_log['altitude_msl'].append(h)
                self.data_log['airspeed'].append(state[0])
                self.data_log['pitch'].append(theta)
                self.data_log['pitch_cmd'].append(pitch_cmd)
                self.data_log['pitch_rate'].append(Q_rate)
                self.data_log['roll'].append(phi)
                self.data_log['heading'].append(heading)
                self.data_log['yaw_cmd'].append(yaw_cmd)
                self.data_log['yaw_rate'].append(R_rate)
                self.data_log['x_position'].append(x)
                self.data_log['y_position'].append(y)
                self.data_log['lateral_offset'].append(lateral_offset)
                self.data_log['longitudinal_offset'].append(longitudinal_offset)
                self.data_log['image_u'].append(estimated_x)
                self.data_log['image_v'].append(estimated_y)
                self.data_log['rudder'].append(rudder)
                self.data_log['elevator'].append(elevator)
                self.data_log['aileron'].append(aileron)
                self.data_log['throttle'].append(throttle)
                self.data_log['target_detected'].append(1 if detected else 0)
                self.data_log['control_mode'].append(self.controller.current_mode)
                self.data_log['gimbal_pitch'].append(gimbal_pitch)
                self.data_log['gimbal_yaw'].append(gimbal_yaw)
                self.data_log['gimbal_locked'].append(1 if self.gimbal.is_locked else 0)
                self.data_log['gimbal_target_pitch'].append(gimbal_target_pitch)
                self.data_log['target_centered'].append(1 if target_centered else 0)
                self.data_log['bbox_width'].append(bbox_width)
                self.data_log['bbox_height'].append(bbox_height)
                self.data_log['vision_fps'].append(self.shared_vision.vision_fps)
                self.data_log['control_fps'].append(ctrl_fps)
                self.data_log['dkf_estimated_u'].append(
                    int(self.dkf.get_position()[0]) if self.dkf is not None and self.dkf.initialized else estimated_x)
                self.data_log['dkf_estimated_v'].append(
                    int(self.dkf.get_position()[1]) if self.dkf is not None and self.dkf.initialized else estimated_y)
                self.data_log['vision_age_ms'].append(vision_age_ms)

                # Raw YOLO only on fresh detections, else NaN
                if is_new_vision and detected:
                    self.data_log['raw_yolo_u'].append(float(raw_center_x))
                    self.data_log['raw_yolo_v'].append(float(raw_center_y))
                else:
                    self.data_log['raw_yolo_u'].append(float('nan'))
                    self.data_log['raw_yolo_v'].append(float('nan'))

                if self.dkf is not None and self.dkf.initialized:
                    dkf_std = self.dkf.get_position_std()
                    self.data_log['dkf_std_u'].append(float(dkf_std[0]))
                    self.data_log['dkf_std_v'].append(float(dkf_std[1]))
                else:
                    self.data_log['dkf_std_u'].append(float('nan'))
                    self.data_log['dkf_std_v'].append(float('nan'))

                self.data_log['is_new_vision'].append(1 if is_new_vision else 0)

                self.data_log['estimated_distance'].append(
                    float(estimated_distance) if estimated_distance is not None else float('nan'))
                self.data_log['distance_confidence'].append(float(distance_confidence))
                self.data_log['trigger_distance'].append(
                    float(self.controller.last_trigger_distance)
                    if self.controller.last_trigger_distance is not None else float('nan'))

                self.data_log['dkf_enabled'].append(1 if self.dkf is not None else 0)
                self.data_log['vision_buffer_delay_ms'].append(
                    float(self.vision_buffer_delay_s * 1000.0))
                self.data_log['use_dkf_for_control'].append(1 if self.use_dkf_for_control else 0)
                self.data_log['vision_drop_rate'].append(float(self.vision_drop_rate))
                self.data_log['vision_dropped'].append(1 if _vision_dropped_this_tick else 0)
                _yolo_raw_hz = self._calc_yolo_rate(self._yolo_raw_arrival_times, window_s=1.0)
                _yolo_eff_hz = self._calc_yolo_rate(self._yolo_effective_arrival_times, window_s=1.0)
                self.data_log['yolo_raw_hz'].append(float(_yolo_raw_hz))
                self.data_log['yolo_effective_hz'].append(float(_yolo_eff_hz))
                self.data_log['ned_fallback_used'].append(int(self._ned_fallback_used_count))

                # TVS diagnostics
                if hasattr(self.controller, 'get_tvs_debug'):
                    _tvs = self.controller.get_tvs_debug()
                else:
                    _tvs = {}
                self.data_log['tvs_active'].append(_tvs.get('tvs_active', 0))
                self.data_log['tvs_beta'].append(float(_tvs.get('tvs_beta', 0.0)))
                self.data_log['tvs_u_norm'].append(float(_tvs.get('tvs_u_norm', 0.0)))
                self.data_log['tvs_v_norm'].append(float(_tvs.get('tvs_v_norm', 0.0)))
                self.data_log['tvs_s_raw'].append(float(_tvs.get('tvs_s_raw', 0.0)))
                self.data_log['tvs_s_ema'].append(float(_tvs.get('tvs_s_ema', 0.0)))
                self.data_log['tvs_corr_pitch_deg'].append(
                    float(_tvs.get('tvs_corr_pitch_deg', 0.0)))
                self.data_log['tvs_corr_heading_deg'].append(
                    float(_tvs.get('tvs_corr_heading_deg', 0.0)))
                self.data_log['tvs_detected'].append(_tvs.get('tvs_detected', 0))
                self.data_log['tvs_edge_flagged'].append(_tvs.get('tvs_edge_flagged', 0))
                self.data_log['tvs_lost_ticks'].append(int(_tvs.get('tvs_lost_ticks', 0)))
                self.data_log['tvs_entered'].append(_tvs.get('tvs_entered', 0))
                self.data_log['tvs_beta_sched'].append(float(_tvs.get('tvs_beta_sched', 0.0)))
                self.data_log['tvs_err_floor'].append(float(_tvs.get('tvs_err_floor', 0.0)))
                self.data_log['tvs_v_norm_ema'].append(float(_tvs.get('tvs_v_norm_ema', 0.0)))
                self.data_log['lead_dt_extrap'].append(float(self._last_lead_dt_extrap))
                self.data_log['lead_set_s'].append(float(self.terminal_lead_s))
                self.data_log['tkf_x'].append(float(self._dbg_tkf_x))
                self.data_log['tkf_y'].append(float(self._dbg_tkf_y))
                self.data_log['tkf_vx'].append(float(self._dbg_tkf_vx))
                self.data_log['tkf_vy'].append(float(self._dbg_tkf_vy))
                self.data_log['tkf_mcount'].append(int(self._dbg_tkf_mcount))

                # ========== 8. Display (low rate) ==========
                control_step += 1

                if show_debug and (control_step % self.DISPLAY_INTERVAL == 0):
                    self._update_display(
                        detected, estimated_x, estimated_y,
                        raw_center_x, raw_center_y,
                        theta, gimbal_pitch, gimbal_yaw,
                        altitude_agl, x, current_mode,
                        estimated_distance, distance_confidence,
                        vision_age_ms, ctrl_fps,
                        gimbal_target_pitch, current_time
                    )

                # ========== 9. Keyboard (low rate) ==========
                if control_step % self.DISPLAY_INTERVAL == 0:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nUser abort")
                        break
                    elif key == ord('r'):
                        print("\nResetting...")
                        self.reset()
                        self.vision_thread.start()
                        time.sleep(0.5)
                        self.pause(False)
                        current_time = 0.0
                        control_step = 0
                        continue
                    elif key == ord('g'):
                        if self.gimbal.is_locked:
                            self.gimbal.unlock_gimbal()
                        else:
                            self.gimbal.lock_gimbal()
                    elif key == ord('w'):
                        self.gimbal.camera_offset_z -= 0.1
                    elif key == ord('s'):
                        self.gimbal.camera_offset_z += 0.1
                    elif key == ord('a'):
                        self.gimbal.camera_offset_x -= 0.1
                    elif key == ord('d'):
                        self.gimbal.camera_offset_x += 0.1
                    elif key == ord('z'):
                        self.gimbal.camera_offset_y += 0.1
                    elif key == ord('x'):
                        self.gimbal.camera_offset_y -= 0.1
                    elif key == ord('0'):
                        self.gimbal.set_camera_position(0.0, 0.0, 0.0)

                current_time += dt

                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nMission interrupted")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.vision_thread.stop()
            self.pause(True)
            cv2.destroyAllWindows()

            # Save data
            if len(self.data_log['time']) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                _kind = getattr(self, 'controller_kind', 'UNK')
                filename = f"RCMQ_Async_V11_{_kind}_{timestamp}.csv"
                filepath = self.log_dir / filename

                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.data_log.keys())
                    rows = zip(*self.data_log.values())
                    writer.writerows(rows)
                print(f"\nData saved: {filepath}")

            print("\n" + "="*70)
            print("Mission end")
            if self.vision_drop_rate > 0:
                actual_drop = (self._drop_count / self._drop_total_eligible
                               if self._drop_total_eligible > 0 else 0.0)
                print(f"Drop stats: set {self.vision_drop_rate*100:.0f}% | "
                      f"actual {self._drop_count}/{self._drop_total_eligible} "
                      f"= {actual_drop*100:.1f}%")
            print("="*70)

    # ==================== Display ====================

    def _update_display(self, detected, est_x, est_y,
                        raw_cx, raw_cy,
                        theta, gimbal_pitch, gimbal_yaw,
                        altitude_agl, x, current_mode,
                        estimated_distance, distance_confidence,
                        vision_age_ms, ctrl_fps,
                        gimbal_target_pitch, current_time):
        """Debug overlay."""
        debug_frame = self.shared_vision.get_debug_frame()
        if debug_frame is None:
            return

        debug_frame = debug_frame.copy()

        # Center cross
        cu, cv_ = int(self.screen_center_u), int(self.screen_center_v)
        cv2.line(debug_frame, (cu - 50, cv_), (cu + 50, cv_), (0, 255, 0), 2)
        cv2.line(debug_frame, (cu, cv_ - 50), (cu, cv_ + 50), (0, 255, 0), 2)

        # SEARCH: zone rectangles
        if current_mode == 'SEARCH':
            half_w = self.screen_width / 2
            half_h = self.screen_height / 2

            green_ratio = self.gimbal.search_green_ratio
            yellow_ratio = self.gimbal.search_yellow_ratio

            gx1 = int(cu - half_w * green_ratio)
            gy1 = int(cv_ - half_h * green_ratio)
            gx2 = int(cu + half_w * green_ratio)
            gy2 = int(cv_ + half_h * green_ratio)
            cv2.rectangle(debug_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

            yx1 = int(cu - half_w * yellow_ratio)
            yy1 = int(cv_ - half_h * yellow_ratio)
            yx2 = int(cu + half_w * yellow_ratio)
            yy2 = int(cv_ + half_h * yellow_ratio)
            cv2.rectangle(debug_frame, (yx1, yy1), (yx2, yy2), (0, 255, 255), 2)

            if detected and self.current_bbox is not None:
                err_x = abs(raw_cx - cu) / half_w if half_w > 0 else 0
                err_y = abs(raw_cy - cv_) / half_h if half_h > 0 else 0
                max_r = max(err_x, err_y)
                if max_r < green_ratio:
                    zone_text = "ZONE: GREEN (normal)"
                    zone_color = (0, 255, 0)
                elif max_r < yellow_ratio:
                    zone_text = "ZONE: GREEN-YELLOW (slow pull)"
                    zone_color = (0, 200, 200)
                else:
                    zone_text = "ZONE: OUTSIDE YELLOW (fast pull)"
                    zone_color = (0, 255, 255)
                cv2.putText(debug_frame, zone_text,
                           (self.screen_width - 450, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)

        # Target display: when the controller consumes the DKF estimate,
        # draw the box at the compensated "now" position (without lead —
        # the leaded aim point runs ahead of the vehicle)
        if detected and self.current_bbox is not None:
            bbox = self.current_bbox

            if self.use_dkf_for_control and self.dkf is not None and self.dkf.initialized:
                disp = getattr(self, '_dkf_display_now', None)
                if disp is not None:
                    gcx, gcy = int(disp[0]), int(disp[1])
                else:
                    gcx, gcy = int(est_x), int(est_y)
                dx = gcx - int(raw_cx)
                dy = gcy - int(raw_cy)
                shifted_bbox = (bbox[0]+dx, bbox[1]+dy, bbox[2]+dx, bbox[3]+dy)
                # Main box (green, thick): target now
                cv2.rectangle(debug_frame,
                              (shifted_bbox[0], shifted_bbox[1]),
                              (shifted_bbox[2], shifted_bbox[3]),
                              (0, 255, 0), 3)
                cv2.circle(debug_frame, (gcx, gcy), 8, (0, 255, 0), -1)
                cv2.putText(debug_frame, "DKF now",
                            (shifted_bbox[0], shifted_bbox[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # Secondary box (dark red, thin): raw delayed YOLO
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (60, 60, 200), 1)
                cv2.circle(debug_frame, (int(raw_cx), int(raw_cy)), 4, (60, 60, 200), -1)
                cv2.putText(debug_frame, f"YOLO -{int(self.vision_buffer_delay_s*1000)}ms",
                            (bbox[2]+5, bbox[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 200), 1)
            else:
                # Raw YOLO bbox (red)
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (255, 0, 0), 3)
                cv2.circle(debug_frame, (int(raw_cx), int(raw_cy)), 6, (0, 0, 255), -1)
        elif not detected and self.enable_dkf and self.dkf is not None and self.dkf.initialized:
            # DKF fill-in during loss (green dot)
            cv2.circle(debug_frame, (int(est_x), int(est_y)), 8, (0, 255, 0), -1)

        mode_color = {
            'SEARCH': (255, 255, 0), 'PRE_DIVE': (255, 165, 0),
            'DIVE': (0, 165, 255)
        }
        color = mode_color.get(current_mode, (0, 165, 255))

        line_of_sight = theta + gimbal_pitch

        y_pos = 50
        display_mode = 'DIVE' if current_mode == 'TERMINAL' else current_mode

        texts = [
            (f'MODE: {display_mode}', color, 1.5, 3),
            (f'Aircraft Pitch: {theta:+.1f}', (0, 255, 255), 0.7, 2),
            (f'Gimbal Pitch: {gimbal_pitch:+.1f} {"[LOCKED]" if self.gimbal.is_locked else ""}',
             (0, 255, 255), 0.7, 2),
            (f'Line of Sight: {line_of_sight:+.1f}', (0, 255, 0), 0.7, 2),
            (f'Vision Delay: +{self.vision_buffer_delay_s*1000:.0f}ms'
             + (' [REAL]' if self.vision_buffer_delay_s > 0 else ' [OFF]'),
             (255, 100, 255) if self.vision_buffer_delay_s > 0 else (150, 150, 150),
             0.7, 2),
            (f'Controller: {"DKF (compensated)" if self.use_dkf_for_control else "raw YOLO"}'
             + (' !' if self.vision_buffer_delay_s > 0 and not self.use_dkf_for_control else ''),
             (0, 255, 0) if self.use_dkf_for_control else (100, 100, 255),
             0.7, 2),
            (f'YOLO raw: {self._calc_yolo_rate(self._yolo_raw_arrival_times, 1.0):.1f}Hz'
             + f'  effective: {self._calc_yolo_rate(self._yolo_effective_arrival_times, 1.0):.1f}Hz'
             + (f'  drop={self.vision_drop_rate*100:.0f}%' if self.vision_drop_rate > 0 else ''),
             (255, 200, 0) if self.vision_drop_rate > 0 else (200, 255, 200),
             0.7, 2),
        ]

        for text, col, scale, thick in texts:
            cv2.putText(debug_frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)
            y_pos += 30 if scale < 1.0 else 50

        # Range info
        trig_dist = self.controller.last_trigger_distance
        if estimated_distance is not None:
            dist_color = (0, 255, 0) if (trig_dist and estimated_distance <= trig_dist) else (0, 200, 255)
            cv2.putText(debug_frame, f'Est Distance: {estimated_distance:.0f}m',
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)
            y_pos += 30

        if trig_dist is not None:
            cv2.putText(debug_frame, f'Trigger Distance: {trig_dist:.0f}m',
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_pos += 30
            if estimated_distance is not None:
                remaining = estimated_distance - trig_dist
                if remaining > 0:
                    cv2.putText(debug_frame, f'Remaining: {remaining:.0f}m',
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(debug_frame, f'TRIGGER READY',
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30

        vision_fps = self.shared_vision.vision_fps
        cv2.putText(debug_frame,
                   f'Vision: {vision_fps:.0f} Hz | Control: {ctrl_fps:.0f} Hz | Age: {vision_age_ms:.0f}ms',
                   (10, self.screen_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.enable_dkf:
            cv2.putText(debug_frame, f'DKF: ON ({1/self.DKF_DT:.0f} Hz predict)',
                       (10, self.screen_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Status line
        gimbal_status = 'L' if self.gimbal.is_locked else 'U'
        dist_str = f"D:{estimated_distance:.0f}m" if estimated_distance else "D:--"

        print(f"\r{current_mode:8s} | "
              f"X:{x:.0f}m | AGL:{altitude_agl:.0f}m | "
              f"th:{theta:+.1f} | G:{gimbal_pitch:+.1f} | "
              f"{dist_str} [{gimbal_status}] | "
              f"V:{vision_fps:.0f}Hz C:{ctrl_fps:.0f}Hz age:{vision_age_ms:.0f}ms",
              end="", flush=True)

        resized = cv2.resize(debug_frame, (1280, 720))
        cv2.imshow('Async Dive Attack', resized)


def main():
    model_path = r'C:\Users\hsu\Desktop\hsu_thesis_simulation\yolo_jeep\runs\detect\yolo_jeep12\weights\best.pt'

    mission = RCMQGimbalDiveAttackAsync(
        model_path=model_path,
        home_heading=176.5,
        local_start=(24306.17578125, -38588.65625),
        start_ground_range=-1000,
        start_elev=201.79 + 300,
        dive_angle=30.0,
        enable_dkf=False,        # required for end-to-end delay experiments
        image_delay=0.0,
        target_position=None,
        # End-to-end delay injection: YOLO held N s before release
        # (0 = off; 0.1/0.2/0.5/0.8 ...)
        vision_buffer_delay_s=0.0,
        # Controller uses DKF estimate (True) or raw YOLO (False).
        # Fly both to compare (control group = False).
        use_dkf_for_control=False,
        # Frame-drop experiment: 0.0 = baseline, 0.3/0.5/0.7 = drop rate
        vision_drop_rate=0.0,
        vision_drop_seed=42,    # fixed seed for identical drop patterns
    )

    mission.controller.set_heading_bias_compensation(1.4)

    mission.run_mission(max_time=180.0, show_debug=True)


if __name__ == '__main__':
    main()