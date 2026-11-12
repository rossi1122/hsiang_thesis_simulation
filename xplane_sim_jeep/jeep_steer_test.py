"""
吉普車轉向測試
==============
測試不同的 DREF 看哪個能讓吉普車轉向。
每種方法持續 8 秒，觀察航向變化。
"""

from xpc3 import XPlaneConnect
import time

xpc = XPlaneConnect()

def read_heading():
    try:
        return xpc.getDREF("sim/flightmodel/position/psi")[0]
    except:
        return -1

def read_speed():
    try:
        return xpc.getDREF("sim/flightmodel/position/groundspeed")[0]
    except:
        return -1

def reset_steering():
    """歸零所有可能的轉向 DREF"""
    try:
        xpc.sendCTRL([0, 0, 0, 0])
        xpc.sendDREF("sim/joystick/yoke_heading_ratio", 0)
        xpc.sendDREF("sim/joystick/yoke_roll_ratio", 0)
        xpc.sendDREF("sim/cockpit2/controls/nosewheel_steer", 0)
        xpc.sendDREF("sim/flightmodel/controls/dist_rat", 0)
        xpc.sendDREF("sim/flightmodel/controls/ail_trim", 0)
    except:
        pass

def run_steer_test(name, steer_fn, duration=8, steer_value=0.5):
    """
    讓車先跑起來，然後套用轉向，觀察航向變化
    """
    print(f"\n{'='*60}")
    print(f"  STEER TEST: {name} (value={steer_value})")
    print(f"{'='*60}")
    
    reset_steering()
    
    # 釋放煞車，給油讓車動
    xpc.sendDREF("sim/flightmodel/controls/parkbrake", 0)
    xpc.sendCTRL([0, 0, 0, 0.5])  # 油門 0.5
    
    # 等車動起來
    print("  Waiting for car to reach speed...")
    time.sleep(3)
    
    h_before = read_heading()
    speed = read_speed()
    print(f"  Heading before: {h_before:.2f}°  Speed: {speed:.2f} m/s")
    
    # 套用轉向
    print(f"  Applying steering for {duration}s...")
    for i in range(duration):
        # 每秒重送
        xpc.sendCTRL([0, 0, 0, 0.5])  # 保持油門
        steer_fn(steer_value)
        time.sleep(1)
        h = read_heading()
        s = read_speed()
        print(f"    t={i+1}s  heading={h:.2f}°  speed={s:.2f} m/s")
    
    h_after = read_heading()
    delta = h_after - h_before
    
    # 停車
    xpc.sendCTRL([0, 0, 0, 0])
    reset_steering()
    xpc.sendDREF("sim/flightmodel/controls/parkbrake", 1)
    time.sleep(1)
    
    result = f"delta={delta:+.2f}°"
    if abs(delta) > 2.0:
        result += "  ✓ STEERING WORKS!"
    else:
        result += "  ✗ No significant turn"
    print(f"  Heading after: {h_after:.2f}°  {result}")
    return delta


results = {}

# --- Test 1: sendCTRL rudder 正值 ---
def steer_ctrl_pos(val):
    xpc.sendCTRL([0, 0, val, 0.5])

results["1. sendCTRL rudder=+0.5"] = run_steer_test(
    "sendCTRL rudder=+0.5", steer_ctrl_pos, steer_value=0.5)

# --- Test 2: sendCTRL rudder 負值 ---
def steer_ctrl_neg(val):
    xpc.sendCTRL([0, 0, -val, 0.5])

results["2. sendCTRL rudder=-0.5"] = run_steer_test(
    "sendCTRL rudder=-0.5", steer_ctrl_neg, steer_value=0.5)

# --- Test 3: yoke_heading_ratio 正 ---
def steer_yoke_heading(val):
    xpc.sendDREF("sim/joystick/yoke_heading_ratio", val)

results["3. yoke_heading_ratio=+0.5"] = run_steer_test(
    "yoke_heading_ratio=+0.5", steer_yoke_heading, steer_value=0.5)

# --- Test 4: yoke_heading_ratio 負 ---
results["4. yoke_heading_ratio=-0.5"] = run_steer_test(
    "yoke_heading_ratio=-0.5", steer_yoke_heading, steer_value=-0.5)

# --- Test 5: yoke_roll_ratio 正 ---
def steer_yoke_roll(val):
    xpc.sendDREF("sim/joystick/yoke_roll_ratio", val)

results["5. yoke_roll_ratio=+0.5"] = run_steer_test(
    "yoke_roll_ratio=+0.5", steer_yoke_roll, steer_value=0.5)

# --- Test 6: yoke_roll_ratio 負 ---
results["6. yoke_roll_ratio=-0.5"] = run_steer_test(
    "yoke_roll_ratio=-0.5", steer_yoke_roll, steer_value=-0.5)

# --- Test 7: nosewheel_steer 正 ---
def steer_nosewheel(val):
    xpc.sendDREF("sim/cockpit2/controls/nosewheel_steer", val)

results["7. nosewheel_steer=+0.5"] = run_steer_test(
    "nosewheel_steer=+0.5", steer_nosewheel, steer_value=0.5)

# --- Test 8: nosewheel_steer 負 ---
results["8. nosewheel_steer=-0.5"] = run_steer_test(
    "nosewheel_steer=-0.5", steer_nosewheel, steer_value=-0.5)

# --- Test 9: 差動煞車（左煞 → 左轉）---
def steer_diff_brake_left(val):
    xpc.sendDREF("sim/cockpit2/controls/left_brake_ratio", val)
    xpc.sendDREF("sim/cockpit2/controls/right_brake_ratio", 0)

results["9. left_brake=0.5 (diff brake)"] = run_steer_test(
    "left_brake=0.5 (differential braking)", steer_diff_brake_left, steer_value=0.5)

# --- Test 10: 差動煞車（右煞 → 右轉）---
def steer_diff_brake_right(val):
    xpc.sendDREF("sim/cockpit2/controls/left_brake_ratio", 0)
    xpc.sendDREF("sim/cockpit2/controls/right_brake_ratio", val)

results["10. right_brake=0.5 (diff brake)"] = run_steer_test(
    "right_brake=0.5 (differential braking)", steer_diff_brake_right, steer_value=0.5)

# --- Test 11: sendCTRL rudder=1.0 (全打) ---
def steer_ctrl_full(val):
    xpc.sendCTRL([0, 0, val, 0.5])

results["11. sendCTRL rudder=+1.0"] = run_steer_test(
    "sendCTRL rudder=+1.0 (full)", steer_ctrl_full, steer_value=1.0)

# --- Test 12: sendCTRL rudder=-1.0 (全打) ---
results["12. sendCTRL rudder=-1.0"] = run_steer_test(
    "sendCTRL rudder=-1.0 (full)", steer_ctrl_full, steer_value=-1.0)


# ═══════════════════════════════════════════════════════
#  結果總結
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print(f"  STEERING TEST RESULTS")
print(f"{'='*60}")
for name, delta in results.items():
    status = "✓ WORKS" if abs(delta) > 2.0 else "✗ FAIL "
    print(f"  {status}  |  {name:45s}  |  delta={delta:+.2f}°")
print(f"{'='*60}")
print(f"\nDone. Please share this output!")
