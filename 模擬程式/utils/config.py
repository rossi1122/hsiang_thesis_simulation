"""
俯衝攻擊系統配置文件
所有可調參數都集中在這裡，方便快速調整
"""

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    # YOLO模型路徑（請根據實際情況修改）
    'model_path': r'C:\Users\hsu\Desktop\hsu_thesis_simulation\yolo\runs\detect\yolo_jeep3\weights\best.pt',
    
    # 檢測參數
    'confidence_threshold': 0.6,    # 信心閾值 [0.3-0.9] 越低越容易檢測，但可能有誤檢
    'max_history': 5,                # 平滑窗口大小 [3-10] 越大越平滑但延遲越大
    'max_lost_frames': 10,           # 目標丟失判定閾值 [5-20] 容忍連續丟失的幀數
    
    # 畫面參數
    'screen_width': 1920,
    'screen_height': 1010,
}

# ==================== 任務配置 ====================
MISSION_CONFIG = {
    # 初始條件
    'initial_altitude': 1500.0,      # 初始高度（米）[500-3000]
    'initial_distance': 3000.0,      # 初始距離目標距離（米）[1000-5000]
    'dive_angle': 45.0,              # 俯衝角度（度）[30-60]
    
    # 安全參數
    'min_attack_altitude': 50.0,     # 最低攻擊高度（米）[30-100]
    
    # 任務參數
    'max_mission_time': 180.0,       # 最大任務時間（秒）
    'control_dt': 0.05,              # 控制週期（秒）[0.03-0.1]
    'show_debug': True,              # 是否顯示調試視窗
}

# ==================== 控制器配置 ====================
CONTROLLER_CONFIG = {
    # 橫向控制PID參數（方向舵 Rudder）
    'lateral_pid': {
        'kp': 0.5,      # 比例增益 [0.1-1.0]
        'ki': 0.05,     # 積分增益 [0.01-0.2]
        'kd': 0.15,     # 微分增益 [0.05-0.3]
    },
    
    # 縱向控制PID參數（升降舵 Elevator - 目標追蹤）
    'longitudinal_pid': {
        'kp': 0.3,      # 比例增益 [0.1-0.8]
        'ki': 0.03,     # 積分增益 [0.01-0.1]
        'kd': 0.1,      # 微分增益 [0.05-0.2]
    },
    
    # 俯仰角控制PID參數（升降舵 Elevator - 角度維持）
    'pitch_pid': {
        'kp': 0.02,     # 比例增益 [0.01-0.05]
        'ki': 0.005,    # 積分增益 [0.001-0.01]
        'kd': 0.015,    # 微分增益 [0.005-0.03]
    },
    
    # 控制限制
    'max_rudder': 1.0,          # 方向舵最大值 [0.5-1.0]
    'max_elevator': 1.0,        # 升降舵最大值 [0.5-1.0]
    'max_aileron': 0.2,         # 副翼輔助最大值 [0.0-0.5]
    
    # 死區參數（避免抖動）
    'lateral_deadzone': 0.05,       # 橫向死區 [0.02-0.1]
    'longitudinal_deadzone': 0.05,  # 縱向死區 [0.02-0.1]
    
    # 控制權重
    'tracking_weight': 0.6,     # 目標追蹤權重 [0.4-0.8]
    'angle_weight': 0.4,        # 角度維持權重 [0.2-0.6]
    # 注意：tracking_weight + angle_weight = 1.0
    
    # 速度控制
    'target_airspeed': 150,     # 目標空速（節）[120-180]
    'max_airspeed': 200,        # 最大允許空速（節）[180-250]
}

# ==================== 調試配置 ====================
DEBUG_CONFIG = {
    # 顯示選項
    'show_fps': True,
    'show_crosshair': True,
    'show_bbox': True,
    'show_offset_line': True,
    'show_flight_info': True,
    
    # 數據記錄
    'enable_logging': True,
    'log_directory': './logs/',
    'save_csv': True,
    
    # 視窗設置
    'window_width': 1280,
    'window_height': 720,
}

# ==================== 預設配置集 ====================
# 可以快速切換不同的配置集

# 保守配置（適合初次測試）
CONSERVATIVE_CONFIG = {
    'lateral_pid': {'kp': 0.3, 'ki': 0.03, 'kd': 0.1},
    'longitudinal_pid': {'kp': 0.2, 'ki': 0.02, 'kd': 0.08},
    'pitch_pid': {'kp': 0.015, 'ki': 0.003, 'kd': 0.01},
    'lateral_deadzone': 0.1,
    'longitudinal_deadzone': 0.1,
}

# 激進配置（適合快速移動目標）
AGGRESSIVE_CONFIG = {
    'lateral_pid': {'kp': 0.7, 'ki': 0.08, 'kd': 0.2},
    'longitudinal_pid': {'kp': 0.5, 'ki': 0.05, 'kd': 0.15},
    'pitch_pid': {'kp': 0.03, 'ki': 0.008, 'kd': 0.02},
    'lateral_deadzone': 0.03,
    'longitudinal_deadzone': 0.03,
}

# 平衡配置（默認推薦）
BALANCED_CONFIG = {
    'lateral_pid': {'kp': 0.5, 'ki': 0.05, 'kd': 0.15},
    'longitudinal_pid': {'kp': 0.3, 'ki': 0.03, 'kd': 0.1},
    'pitch_pid': {'kp': 0.02, 'ki': 0.005, 'kd': 0.015},
    'lateral_deadzone': 0.05,
    'longitudinal_deadzone': 0.05,
}

# ==================== 場景配置 ====================
# 不同場景的推薦參數

# 場景1：高空俯衝（1500m+）
HIGH_ALTITUDE_DIVE = {
    'initial_altitude': 1500.0,
    'initial_distance': 3000.0,
    'dive_angle': 45.0,
    'tracking_weight': 0.5,
    'angle_weight': 0.5,  # 高空更注重角度維持
}

# 場景2：中空俯衝（800m）
MID_ALTITUDE_DIVE = {
    'initial_altitude': 800.0,
    'initial_distance': 2000.0,
    'dive_angle': 40.0,
    'tracking_weight': 0.6,
    'angle_weight': 0.4,
}

# 場景3：低空俯衝（400m）
LOW_ALTITUDE_DIVE = {
    'initial_altitude': 400.0,
    'initial_distance': 1000.0,
    'dive_angle': 35.0,
    'tracking_weight': 0.7,
    'angle_weight': 0.3,  # 低空更注重目標追蹤
}

# 場景4：陡峭俯衝（60度）
STEEP_DIVE = {
    'initial_altitude': 1500.0,
    'initial_distance': 2500.0,
    'dive_angle': 60.0,
    'tracking_weight': 0.5,
    'angle_weight': 0.5,
    'target_airspeed': 180,  # 陡峭俯衝速度更快
}

# ==================== 使用說明 ====================
"""
使用方法：

1. 基本使用（使用默認配置）：
   python car_dive_attack_main.py

2. 使用預設配置集：
   在 car_dive_attack_main.py 中：
   
   from config import CONTROLLER_CONFIG, AGGRESSIVE_CONFIG
   CONTROLLER_CONFIG.update(AGGRESSIVE_CONFIG)

3. 使用場景配置：
   
   from config import MISSION_CONFIG, LOW_ALTITUDE_DIVE
   MISSION_CONFIG.update(LOW_ALTITUDE_DIVE)

4. 自定義配置：
   
   直接修改此文件中的參數值

參數調整建議：

1. 如果目標追蹤不夠靈敏：
   - 增加 lateral_kp 和 longitudinal_kp
   - 減小 lateral_deadzone 和 longitudinal_deadzone
   - 增加 tracking_weight

2. 如果控制震蕩：
   - 降低 kp 值
   - 增加 kd 值
   - 增加死區值

3. 如果無法維持俯衝角度：
   - 增加 pitch_kp
   - 增加 angle_weight
   - 降低 tracking_weight

4. 如果反應太慢：
   - 減小 control_dt
   - 增加所有 kp 和 kd 值

5. 如果檢測不穩定：
   - 降低 confidence_threshold
   - 增加 max_history
   - 增加 max_lost_frames
"""

# ==================== 導出配置 ====================
def get_default_config():
    """獲取默認配置"""
    return {
        'model': MODEL_CONFIG,
        'mission': MISSION_CONFIG,
        'controller': CONTROLLER_CONFIG,
        'debug': DEBUG_CONFIG,
    }

def get_config_by_name(config_name):
    """
    根據名稱獲取配置
    
    Args:
        config_name: 配置名稱
            - 'conservative': 保守配置
            - 'aggressive': 激進配置
            - 'balanced': 平衡配置（默認）
            - 'high_altitude': 高空俯衝
            - 'mid_altitude': 中空俯衝
            - 'low_altitude': 低空俯衝
            - 'steep': 陡峭俯衝
    
    Returns:
        config: 配置字典
    """
    base_config = get_default_config()
    
    config_map = {
        'conservative': CONSERVATIVE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG,
        'balanced': BALANCED_CONFIG,
        'high_altitude': HIGH_ALTITUDE_DIVE,
        'mid_altitude': MID_ALTITUDE_DIVE,
        'low_altitude': LOW_ALTITUDE_DIVE,
        'steep': STEEP_DIVE,
    }
    
    if config_name in config_map:
        if 'pid' in str(config_map[config_name]):
            base_config['controller'].update(config_map[config_name])
        else:
            base_config['mission'].update(config_map[config_name])
    
    return base_config

def print_config_summary(config):
    """打印配置摘要"""
    print("\n" + "=" * 60)
    print("當前配置摘要")
    print("=" * 60)
    
    print("\n【任務配置】")
    print(f"  初始高度: {config['mission']['initial_altitude']}m")
    print(f"  俯衝角度: {config['mission']['dive_angle']}°")
    print(f"  初始距離: {config['mission']['initial_distance']}m")
    
    print("\n【控制器配置】")
    lateral = config['controller']['lateral_pid']
    print(f"  橫向PID: Kp={lateral['kp']}, Ki={lateral['ki']}, Kd={lateral['kd']}")
    
    longitudinal = config['controller']['longitudinal_pid']
    print(f"  縱向PID: Kp={longitudinal['kp']}, Ki={longitudinal['ki']}, Kd={longitudinal['kd']}")
    
    print(f"  追蹤權重: {config['controller']['tracking_weight']}")
    print(f"  角度權重: {config['controller']['angle_weight']}")
    
    print("\n【檢測配置】")
    print(f"  信心閾值: {config['model']['confidence_threshold']}")
    print(f"  平滑窗口: {config['model']['max_history']}")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # 測試配置
    print("測試配置系統...")
    
    # 默認配置
    default_config = get_default_config()
    print_config_summary(default_config)
    
    # 激進配置
    print("\n切換到激進配置...")
    aggressive_config = get_config_by_name('aggressive')
    print_config_summary(aggressive_config)
    
    # 低空俯衝配置
    print("\n切換到低空俯衝配置...")
    low_alt_config = get_config_by_name('low_altitude')
    print_config_summary(low_alt_config)