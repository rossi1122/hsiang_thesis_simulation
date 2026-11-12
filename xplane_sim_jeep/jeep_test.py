"""
吉普車控制方法測試腳本
======================
逐一測試不同的 DREF 組合，找出哪個能讓吉普車動起來。
每種方法測試 5 秒，印出速度變化。

使用方式：
  1. 在 X-Plane 中把吉普車放在跑道上
  2. python jeep_test.py
  3. 觀察哪個方法讓車動了
"""

from xpc3 import XPlaneConnect
import time

xpc = XPlaneConnect()

def read_speed():
    try:
        return xpc.getDREF("sim/flightmodel/position/groundspeed")[0]
    except:
        return -1

def reset():
    """每次測試前歸零"""
    try:
        xpc.sendDREF("sim/flightmodel/engine/ENGN_thro", 0)
        xpc.sendDREF("sim/flightmodel/engine/ENGN_thro_use", 0)
        xpc.sendDREF("sim/flightmodel/controls/parkbrake", 1)
        xpc.sendDREF("sim/operation/override/override_joystick", 0)
        xpc.sendDREF("sim/operation/override/override_throttles", 0)
        xpc.sendCTRL([0, 0, 0, 0])
        xpc.sendDREF("sim/joystick/yoke_heading_ratio", 0)
        xpc.sendDREF("sim/joystick/yoke_roll_ratio", 0)
        xpc.sendDREF("sim/joystick/yoke_pitch_ratio", 0)
        xpc.sendDREF("sim/cockpit2/engine/actuators/throttle_ratio_all", 0)
    except:
        pass
    time.sleep(1)

def run_test(name, setup_fn, duration=5):
    """執行一個測試方法"""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    
    reset()
    
    # 釋放手煞車
    xpc.sendDREF("sim/flightmodel/controls/parkbrake", 0)
    time.sleep(0.3)
    
    speed_before = read_speed()
    print(f"  Speed before: {speed_before:.4f} m/s")
    
    # 執行測試設定
    setup_fn()
    
    # 持續送指令 5 秒，每秒印速度
    print(f"  Running for {duration}s...")
    for i in range(duration):
        setup_fn()  # 每秒重送一次
        time.sleep(1)
        speed = read_speed()
        print(f"    t={i+1}s  speed={speed:.4f} m/s")
    
    speed_after = read_speed()
    moved = speed_after > 0.1
    result = "✓ MOVED!" if moved else "✗ No movement"
    print(f"  Speed after: {speed_after:.4f} m/s  --> {result}")
    
    reset()
    return moved


# ═══════════════════════════════════════════════════════
#  測試方法
# ═══════════════════════════════════════════════════════

results = {}

# --- Test 1: sendCTRL only ---
def test1():
    xpc.sendCTRL([0, 0, 0, 0.8])

results["1. sendCTRL throttle=0.8"] = run_test(
    "sendCTRL([0, 0, 0, 0.8])", test1)

# --- Test 2: override_joystick + sendCTRL ---
def test2():
    xpc.sendDREF("sim/operation/override/override_joystick", 1)
    xpc.sendCTRL([0, 0, 0, 0.8])

results["2. override_joystick + sendCTRL"] = run_test(
    "override_joystick=1, sendCTRL throttle=0.8", test2)

# --- Test 3: ENGN_thro only (no override) ---
def test3():
    xpc.sendDREF("sim/flightmodel/engine/ENGN_thro", 0.8)

results["3. ENGN_thro only"] = run_test(
    "ENGN_thro=0.8 (no override)", test3)

# --- Test 4: override_throttles + ENGN_thro ---
def test4():
    xpc.sendDREF("sim/operation/override/override_throttles", 1)
    xpc.sendDREF("sim/flightmodel/engine/ENGN_thro", 0.8)

results["4. override_throttles + ENGN_thro"] = run_test(
    "override_throttles=1, ENGN_thro=0.8", test4)

# --- Test 5: ENGN_thro_use ---
def test5():
    xpc.sendDREF("sim/flightmodel/engine/ENGN_thro_use", 0.8)

results["5. ENGN_thro_use only"] = run_test(
    "ENGN_thro_use=0.8", test5)

# --- Test 6: throttle_ratio_all ---
def test6():
    xpc.sendDREF("sim/cockpit2/engine/actuators/throttle_ratio_all", 0.8)

results["6. throttle_ratio_all"] = run_test(
    "throttle_ratio_all=0.8", test6)

# --- Test 7: override_joystick + ENGN_thro ---
def test7():
    xpc.sendDREF("sim/operation/override/override_joystick", 1)
    xpc.sendDREF("sim/flightmodel/engine/ENGN_thro", 0.8)

results["7. override_joystick + ENGN_thro"] = run_test(
    "override_joystick=1, ENGN_thro=0.8", test7)

# --- Test 8: throttle_beta_rev ---
def test8():
    xpc.sendDREF("sim/cockpit2/engine/actuators/throttle_beta_rev_ratio_all", 0.8)

results["8. throttle_beta_rev_ratio_all"] = run_test(
    "throttle_beta_rev_ratio_all=0.8", test8)

# --- Test 9: sendCTRL without any override ---
def test9_setup():
    xpc.sendDREF("sim/operation/override/override_joystick", 0)
    xpc.sendDREF("sim/operation/override/override_throttles", 0)

def test9():
    xpc.sendCTRL([0, 0, 0, 0.8])

results["9. sendCTRL no override"] = run_test(
    "No override, sendCTRL throttle=0.8", test9)

# ═══════════════════════════════════════════════════════
#  結果總結
# ═══════════════════════════════════════════════════════

print(f"\n\n{'='*60}")
print(f"  RESULTS SUMMARY")
print(f"{'='*60}")
for name, moved in results.items():
    status = "✓ WORKS" if moved else "✗ FAIL"
    print(f"  {status}  |  {name}")
print(f"{'='*60}")
print(f"\nDone. Please share this output!")
