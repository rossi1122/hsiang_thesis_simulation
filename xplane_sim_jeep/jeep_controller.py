"""
X-Plane 11 吉普車自動駕駛控制器（含 CSV 記錄）
================================================


使用方式：
  1. 在吉普車的 X-Plane 中，先把車放在跑道上的起始位置
  2. python jeep_controller.py
  3. 按 Ctrl+C 停車（CSV 自動存檔）
"""

import time
import math
import sys
import csv
import os
import argparse
from datetime import datetime

from xpc3 import XPlaneConnect


# ═══════════════════════════════════════════════════════
#  設定參數（可透過命令列覆蓋）
# ═══════════════════════════════════════════════════════

DEFAULT_TARGET_SPEED = 10.0       # 目標速度 (m/s)
DEFAULT_TARGET_HEADING = 177   # 目標航向 (度, 真北順時針)
DEFAULT_CONTROL_HZ = 20         # 控制迴圈頻率 (Hz)
DEFAULT_MAX_DURATION = 300       # 最長行駛時間 (秒)
LOG_DIR = "jeep_logs"            # CSV 存放資料夾

#Home frame 座標轉換參數（與飛機端一致）
DEFAULT_LOCAL_START = (24306.17578125, -38588.65625)
DEFAULT_ROTATION_ANGLE = -3.327831058162679  # degrees


class JeepController:
    """
    吉普車自動駕駛控制器（含即時 CSV 記錄）
    
   新增 home frame 座標（與飛機共用同一個 local_start + 旋轉角）
    """
    
    def __init__(self, 
                 target_speed=DEFAULT_TARGET_SPEED,
                 target_heading=DEFAULT_TARGET_HEADING,
                 control_hz=DEFAULT_CONTROL_HZ,
                 xp_host='localhost',
                 xp_port=49009,
                 local_start=DEFAULT_LOCAL_START,
                 rotation_angle=DEFAULT_ROTATION_ANGLE):
        self.target_speed = target_speed
        self.target_heading = target_heading
        self.control_dt = 1.0 / control_hz
        self.control_hz = control_hz
        
        # ── Home frame 座標轉換參數 ──
        self._local_start = local_start
        self._rotation_angle = rotation_angle
        
        # ── 航向 PID 控制參數 ──
        self.heading_Kp = 0.015
        self.heading_Ki = 0.005
        self.heading_Kd = 0.005
        self.max_steer = 0.15
        
        # ── 速度 PI 控制參數 ──
        self.speed_Kp = 0.4
        self.speed_Ki = 0.10
        self.throttle_base = 0.0
        self.throttle_min = 0.0
        self.throttle_max = 1.0
        self.brake_Kp = 0.5
        
        # ── 控制狀態 ──
        self._prev_heading_err = 0.0
        self._heading_integral = 0.0
        self._speed_integral = 0.0
        
        # ── CSV 記錄 ──
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        
        # ── 連接 X-Plane ──
        print(f"\n{'='*60}")
        print(f"  吉普車自動駕駛控制器 V2")
        print(f"{'='*60}")
        print(f"  目標速度:   {target_speed:.1f} m/s ({target_speed*3.6:.1f} km/h)")
        print(f"  目標航向:   {target_heading:.1f}°")
        print(f"  控制頻率:   {control_hz} Hz")
        print(f"  XPC 連接:   {xp_host}:{xp_port}")
        print(f"  ✨ Home frame: local_start={local_start}")
        print(f"               rotation={rotation_angle:.6f}°")
        print(f"{'='*60}\n")
        
        print("[XPC] 正在連接 X-Plane...")
        try:
            self.client = XPlaneConnect(xpHost=xp_host, xpPort=xp_port)
            print("[XPC] ✓ 連接成功\n")
        except Exception as e:
            print(f"[XPC] ✗ 連接失敗: {e}")
            sys.exit(1)
    
    # ──  Home frame 座標轉換 ────────────────────
    # （與飛機端 car_dive_attack_gimbal_v5_async.py 完全一致）
    
    def _rotate_to_home(self, y, x):
        """將 local 偏移量旋轉到 home frame"""
        a = math.cos(math.radians(self._rotation_angle))
        b = math.sin(math.radians(self._rotation_angle))
        return a * y - b * x, b * y + a * x
    
    def _local_to_home(self, local_z, local_x):
        """
        X-Plane local 座標 → home frame (x, y)
        
        Args:
            local_z: sim/flightmodel/position/local_z
            local_x: sim/flightmodel/position/local_x
        
        Returns:
            (x_home, y_home): home frame 座標
              x_home: 沿跑道方向（負值=在目標前方）
              y_home: 橫向偏移
        """
        start_Z, start_X = self._local_start
        dz = local_z - start_Z
        dx = local_x - start_X
        rotx, roty = self._rotate_to_home(dz, dx)
        return rotx, -roty
    
    def get_home_xy(self, local_z, local_x):
        """便捷方法：從 state dict 的 local_z, local_x 算出 home (x, y)"""
        return self._local_to_home(local_z, local_x)
    
    # ── 讀取狀態 ──────────────────────────────────────
    
    def get_state(self):
        try:
            heading = self.client.getDREF("sim/flightmodel/position/psi")[0]
            groundspeed = self.client.getDREF("sim/flightmodel/position/groundspeed")[0]
            lat = self.client.getDREF("sim/flightmodel/position/latitude")[0]
            lon = self.client.getDREF("sim/flightmodel/position/longitude")[0]
            local_x = self.client.getDREF("sim/flightmodel/position/local_x")[0]
            local_z = self.client.getDREF("sim/flightmodel/position/local_z")[0]
            
            # 計算 home frame 座標
            x_pos, y_pos = self.get_home_xy(local_z, local_x)
            
            return {
                'heading': heading, 'groundspeed': groundspeed,
                'lat': lat, 'lon': lon,
                'local_x': local_x, 'local_z': local_z,
                'x_position': x_pos, 'y_position': y_pos, 
            }
        except Exception as e:
            print(f"[讀取] 錯誤: {e}")
            return None
    
    # ── 控制計算 ──────────────────────────────────────
    
    @staticmethod
    def heading_error(current, target):
        err = target - current
        while err > 180:
            err -= 360
        while err < -180:
            err += 360
        return err
    
    def compute_rudder(self, heading_err):
        p_term = self.heading_Kp * heading_err
        self._heading_integral += heading_err * self.control_dt
        self._heading_integral = max(-50.0, min(50.0, self._heading_integral))
        i_term = self.heading_Ki * self._heading_integral
        d_term = self.heading_Kd * (heading_err - self._prev_heading_err) / self.control_dt
        self._prev_heading_err = heading_err
        
        rudder = p_term + i_term + d_term
        return max(-self.max_steer, min(self.max_steer, rudder))
    
    def compute_throttle_and_brake(self, speed_err):
        """
        速度控制：三段式（加油 / 滑行 / 煞車）
        """
        DEADBAND = 0.3
        
        if speed_err > 0:
            p_term = self.speed_Kp * speed_err
            self._speed_integral += speed_err * self.control_dt
            self._speed_integral = max(0, min(5.0, self._speed_integral))
            i_term = self.speed_Ki * self._speed_integral
            throttle = self.throttle_base + p_term + i_term
            throttle = max(0.0, min(self.throttle_max, throttle))
            return throttle, 0.0
        elif abs(speed_err) <= DEADBAND:
            self._speed_integral *= 0.99
            return 0.0, 0.0
        else:
            self._speed_integral *= 0.9
            overspeed = abs(speed_err) - DEADBAND
            brake = min(1.0, self.brake_Kp * overspeed)
            return 0.0, brake
    
    # ── CSV 記錄 ──────────────────────────────────────
    
    def _open_csv(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(LOG_DIR, f"jeep_{timestamp}.csv")
        self._csv_file = open(self._csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
      
        self._csv_writer.writerow([
            'wall_time',                  
            'time', 'heading', 'groundspeed',
            'lat', 'lon', 'local_x', 'local_z',
            'x_position', 'y_position',   
            'heading_err', 'rudder', 'throttle'
        ])
        print(f"[CSV] 記錄檔: {self._csv_path}")
    
    def _write_csv_row(self, elapsed, state, h_err, rudder, throttle, wall_time=None):
        if self._csv_writer:
            if wall_time is None:
                wall_time = time.time()
            self._csv_writer.writerow([
                f"{wall_time:.6f}",              
                f"{elapsed:.3f}",
                f"{state['heading']:.4f}",
                f"{state['groundspeed']:.4f}",
                f"{state['lat']:.8f}",
                f"{state['lon']:.8f}",
                f"{state['local_x']:.4f}",
                f"{state['local_z']:.4f}",
                f"{state['x_position']:.4f}",    
                f"{state['y_position']:.4f}",    
                f"{h_err:.4f}",
                f"{rudder:.6f}",
                f"{throttle:.6f}",
            ])
    
    def _close_csv(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            print(f"[CSV] ✓ 已存檔: {self._csv_path}")
    
    # ── 初始化 / 停車 ────────────────────────────────
    
    def initialize(self):
        print("[初始化] 釋放手煞車...")
        self.client.sendDREF("sim/flightmodel/controls/parkbrake", 0)
        self.client.sendDREF("sim/cockpit2/controls/left_brake_ratio", 0)
        self.client.sendDREF("sim/cockpit2/controls/right_brake_ratio", 0)
        self.client.sendCTRL([0, 0, 0, 0])
        self.client.sendDREF("sim/joystick/yoke_roll_ratio", 0)
        time.sleep(0.5)
        
        state = self.get_state()
        if state:
            print(f"[初始化] 當前航向: {state['heading']:.1f}°")
            print(f"[初始化] 當前速度: {state['groundspeed']:.2f} m/s")
            print(f"[初始化] 當前位置: ({state['lat']:.6f}, {state['lon']:.6f})")
            print(f"[初始化] Home frame: x={state['x_position']:.1f}m, y={state['y_position']:.1f}m")
        print("[初始化] ✓ 就緒\n")
    
    def stop(self):
        print("\n[停車] 正在停車...")
        try:
            self.client.sendCTRL([0, 0, 0, 0])
            self.client.sendDREF("sim/joystick/yoke_roll_ratio", 0)
            self.client.sendDREF("sim/cockpit2/controls/left_brake_ratio", 1.0)
            self.client.sendDREF("sim/cockpit2/controls/right_brake_ratio", 1.0)
            time.sleep(0.5)
            self.client.sendDREF("sim/flightmodel/controls/parkbrake", 1)
            self.client.sendDREF("sim/cockpit2/controls/left_brake_ratio", 0)
            self.client.sendDREF("sim/cockpit2/controls/right_brake_ratio", 0)
            print("[停車] ✓ 已停車並釋放控制")
        except Exception as e:
            print(f"[停車] 錯誤: {e}")
    
    # ── 主控制迴圈 ────────────────────────────────────
    
    def run(self, max_duration=DEFAULT_MAX_DURATION):
        self.initialize()
        self._open_csv()
        
        start_time = time.time()
        loop_count = 0
        print_interval = int(1.0 / self.control_dt)
        
        print(f"[行駛] 開始！目標: {self.target_speed:.1f} m/s @ {self.target_heading:.1f}°")
        print(f"[行駛] 按 Ctrl+C 停車\n")
        print(f"{'時間':>6s} | {'航向':>7s} | {'偏差':>6s} | {'速度':>8s} | {'X(home)':>8s} | {'Y(home)':>8s} | {'Rudder':>7s} | {'Throttle':>8s}")
        print(f"{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
        
        try:
            while True:
                loop_start = time.time()
                elapsed = loop_start - start_time
                
                if elapsed > max_duration:
                    print(f"\n[行駛] 已達最大行駛時間 ({max_duration}s)，自動停車")
                    break
                
                state = self.get_state()
                if state is None:
                    time.sleep(self.control_dt)
                    continue
                
                heading = state['heading']
                speed = state['groundspeed']
                
                h_err = self.heading_error(heading, self.target_heading)
                rudder = self.compute_rudder(h_err)
                s_err = self.target_speed - speed
                throttle, brake = self.compute_throttle_and_brake(s_err)
                
                self.client.sendCTRL([0, 0, 0, throttle])
                self.client.sendDREF("sim/joystick/yoke_roll_ratio", rudder)
                self.client.sendDREF("sim/cockpit2/controls/left_brake_ratio", brake)
                self.client.sendDREF("sim/cockpit2/controls/right_brake_ratio", brake)
                
                self._write_csv_row(elapsed, state, h_err, rudder, throttle,
                                    wall_time=loop_start)
                
                loop_count += 1
                if loop_count % print_interval == 0:
                    print(f"{elapsed:6.1f} | {heading:7.1f}° | {h_err:+6.1f}° | "
                          f"{speed:5.2f} m/s | {state['x_position']:+8.1f} | {state['y_position']:+8.1f} | "
                          f"{rudder:+7.3f} | {throttle:8.3f}")
                
                loop_elapsed = time.time() - loop_start
                sleep_time = self.control_dt - loop_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n\n[行駛] 收到 Ctrl+C，準備停車...")
        finally:
            self.stop()
            self._close_csv()
        
        total_time = time.time() - start_time
        print(f"[完成] 總行駛時間: {total_time:.1f} 秒")
        print(f"[提示] 執行 python jeep_plot.py 即可繪圖\n")


def main():
    parser = argparse.ArgumentParser(
        description="X-Plane 11 吉普車自動駕駛控制器 V2（含 home frame 座標）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python jeep_controller.py                        # 預設 5 m/s, 航向 179.5°
  python jeep_controller.py --speed 8              # 8 m/s
  python jeep_controller.py --speed 3 --heading 0  # 3 m/s 往北開
  python jeep_controller.py --speed 15             # 15 m/s (54 km/h)

停車後繪圖：
  python jeep_plot.py                              # 自動抓最新 CSV
  python thesis_plot_moving_target.py              # 自動抓飛機+Jeep最新CSV
        """
    )
    parser.add_argument('--speed', type=float, default=DEFAULT_TARGET_SPEED,
                        help=f'目標速度 m/s (預設: {DEFAULT_TARGET_SPEED})')
    parser.add_argument('--heading', type=float, default=DEFAULT_TARGET_HEADING,
                        help=f'目標航向 度 (預設: {DEFAULT_TARGET_HEADING})')
    parser.add_argument('--hz', type=int, default=DEFAULT_CONTROL_HZ,
                        help=f'控制頻率 Hz (預設: {DEFAULT_CONTROL_HZ})')
    parser.add_argument('--duration', type=float, default=DEFAULT_MAX_DURATION,
                        help=f'最大行駛時間 秒 (預設: {DEFAULT_MAX_DURATION})')
    parser.add_argument('--host', type=str, default='localhost',
                        help='X-Plane 主機 IP (預設: localhost)')
    parser.add_argument('--port', type=int, default=49009,
                        help='XPC plugin port (預設: 49009)')
    
    args = parser.parse_args()
    
    controller = JeepController(
        target_speed=args.speed,
        target_heading=args.heading,
        control_hz=args.hz,
        xp_host=args.host,
        xp_port=args.port,
    )
    controller.run(max_duration=args.duration)


if __name__ == '__main__':
    main()