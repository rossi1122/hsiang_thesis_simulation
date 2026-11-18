"""座標系、針孔投影、Ideal State 轉換（Eq. 4–8、15、54）。

座標慣例（全部右手系）
- 世界座標：x、y 在水平面，z 朝上。航向 ψ 從 +x 量起、逆時針為正，
  所以 Eq. 1 的 ẋ = V cosψ、ẏ = V sinψ、ψ̇ = u_ψ 直接成立；u_ψ < 0 是右轉。
- 機體座標 FLU：x 機頭、y 左翼、z 朝上。
- 雲台：pan θ_p 繞機體 z 軸、往左為正；tilt θ_t 是仰角、往下為負。
  Ideal State (Eq. 7) θ_p = -π/2（朝右側）、θ_t = -α（往下 α）。
- 相機座標：z_c 光軸、x_c 影像右（u）、y_c 影像下（v），u = f·x_c/z_c、v = f·y_c/z_c。

用這組慣例推導，相機以 V_t 前進、以 u_ψ 偏航時，標準影像雅可比 L_s (Eq. 6)
會得到和論文 Eq. 8 完全相同的式子（tests/test_geometry.py 用數值微分驗證）。
"""
import numpy as np


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_to_world(psi, pitch=0.0, roll=0.0):
    """機體 FLU → 世界。pitch 機頭上仰為正，roll 右翼下沉為正。"""
    return rot_z(psi) @ rot_y(-pitch) @ rot_x(roll)


def camera_rows(pan, tilt):
    """回傳 3×3 矩陣，三列依序是 x_c、y_c、z_c 在機體座標的方向。"""
    optical = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)])
    right = np.array([np.sin(pan), -np.cos(pan), 0.0])
    down = np.cross(optical, right)
    return np.vstack([right, down, optical])


def project(target, uav, psi, pan, tilt, f, pitch=0.0, roll=0.0):
    """目標世界座標 → 真實相機特徵 (u1, v1)。回傳 (u, v, 深度)；深度 <= 0 表示在相機後方。"""
    rel = np.asarray(target, float) - np.asarray(uav, float)
    body = body_to_world(psi, pitch, roll).T @ rel
    cam = camera_rows(pan, tilt) @ body
    return f * cam[0] / cam[2], f * cam[1] / cam[2], cam[2]


def pointing_angles(target, uav, psi, pitch=0.0, roll=0.0):
    """讓光軸正對目標所需的 (pan, tilt)。"""
    rel = np.asarray(target, float) - np.asarray(uav, float)
    b = body_to_world(psi, pitch, roll).T @ rel
    return np.arctan2(b[1], b[0]), np.arctan2(b[2], np.hypot(b[0], b[1]))


def alpha_from(H, R_d):
    """Eq. 7：α = atan(H / R_d)。"""
    return np.arctan2(H, R_d)


def to_ideal(u1, v1, pan, tilt, alpha, f, pitch=0.0, roll=0.0):
    """Eq. 7：真實相機特徵 (u1, v1) → Ideal State 特徵 (u2, v2)。

    做法同前作（Yang 2021 JIRS, Eq. 3–12）：把雲台與飛機的俯仰、滾轉轉掉，
    只保留航向，再轉進 θ_p = -π/2、θ_t = -α 的虛擬相機。純旋轉，不需要深度。
    """
    ray_body = camera_rows(pan, tilt).T @ np.array([u1, v1, f])
    ray_level = rot_y(-pitch) @ rot_x(roll) @ ray_body  # 只去掉俯仰、滾轉，航向保留
    r = camera_rows(-np.pi / 2, -alpha) @ ray_level
    return f * r[0] / r[2], f * r[1] / r[2]


def ideal_and_down(u1, v1, pan, tilt, alpha, f, pitch=0.0, roll=0.0, min_cos=0.01):
    """一次算出 Ideal State (u2, v2)、垂直相機 (u3, v3)，並判斷 (u2, v2) 是否有效。

    Ideal 相機朝右側看；目標在飛機左側超過 H·tanα（= H²/R_d = 200 m）時會跑到這台
    虛擬相機後方，(u2, v2) 沒有定義（論文沒處理）。valid = 與光軸夾角 < arccos(min_cos)。
    (u3, v3) 直接由射線算：垂直朝下相機對地面目標永遠有效，有效時與 Eq. 54 相同。
    """
    ray_body = camera_rows(pan, tilt).T @ np.array([u1, v1, f])
    ray_level = rot_y(-pitch) @ rot_x(roll) @ ray_body
    r2 = camera_rows(-np.pi / 2, -alpha) @ ray_level
    r3 = camera_rows(-np.pi / 2, -np.pi / 2) @ ray_level
    valid = r2[2] > min_cos * np.linalg.norm(r2)
    u2 = f * r2[0] / r2[2] if valid else np.nan
    v2 = f * r2[1] / r2[2] if valid else np.nan
    return u2, v2, f * r3[0] / r3[2], f * r3[1] / r3[2], bool(valid)


# ---------------------------------------------------------------------------
# 一般化的相機姿態（2026-11-18，增穩雲台用）
# 相機姿態用 3×3「相機列」矩陣 C 表示：三列依序是 x_c（影像右）、y_c（影像下）、z_c（光軸）。
#   C_body：在機體座標；C_H：在「水平航向座標」H（x = 機頭方向的水平投影、y 左、z 上，也就是世界座標轉掉 ψ）。
#   Ideal State（Eq. 7）就是在 H 座標裡定義的虛擬相機，所以量測換算只需要 C_H。
# 原本的 camera_rows(pan, tilt) 是「兩軸、相對機身」的特例：C_body = camera_rows(pan, tilt)。
# ---------------------------------------------------------------------------
M0 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # 所有角度為 0 時的相機列（機體座標）


def level_from_body(pitch, roll):
    """機體座標 → 水平航向座標 H 的旋轉（只去掉俯仰、滾轉）。"""
    return rot_y(-pitch) @ rot_x(roll)


def rows_body_to_level(C_body, pitch, roll):
    return C_body @ level_from_body(pitch, roll).T


def rows_level_to_body(C_H, pitch, roll):
    return C_H @ level_from_body(pitch, roll)


def rows_from_joints(yaw, roll, pitch_up):
    """雲台關節（底座往外：yaw 繞機體 z → roll 繞 x → pitch 繞 y）→ 機體座標相機列。

    pitch_up 往上為正（SIYI 的慣例，−90° = 垂直朝下）；roll 往右為正。
    roll = 0 時等於 camera_rows(yaw, pitch_up)。
    """
    J = rot_z(yaw) @ rot_x(roll) @ rot_y(-pitch_up)
    return M0 @ J.T


def joints_from_rows(C_body):
    """rows_from_joints 的反函數（ZXY 分解，取 |roll| ≤ 90° 的那一組）。回傳 (yaw, roll, pitch_up)。"""
    J = C_body.T @ M0
    roll = float(np.arcsin(np.clip(J[2, 1], -1.0, 1.0)))
    yaw = float(np.arctan2(-J[0, 1], J[1, 1]))
    pitch_down = float(np.arctan2(-J[2, 0], J[2, 2]))
    return yaw, roll, -pitch_down


def head_angles_from_rows(C_body):
    """機體座標相機列 → X-Plane 頭部視角 (偏航, 俯仰往上為正, 滾轉)，ZYX 順序（先偏航、再俯仰、最後繞視線滾轉）。

    兩軸機身雲台時剛好是 (pan, tilt, 0)，和原本送給 X-Plane 的值相同。
    """
    R = C_body.T @ M0
    pitch_down = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return yaw, -pitch_down, roll


def ray_level_from_pixel(u1, v1, C_H, f):
    """像素 → 水平航向座標 H 中的射線方向（未正規化）。"""
    return C_H.T @ np.array([u1, v1, f], float)


def ideal_and_down_from_ray(ray_level, alpha, f, min_cos=0.01):
    """H 座標射線 → Ideal State (u2, v2)、垂直相機 (u3, v3)、valid（同 ideal_and_down）。"""
    r2 = camera_rows(-np.pi / 2, -alpha) @ ray_level
    r3 = camera_rows(-np.pi / 2, -np.pi / 2) @ ray_level
    valid = r2[2] > min_cos * np.linalg.norm(r2)
    u2 = f * r2[0] / r2[2] if valid else np.nan
    v2 = f * r2[1] / r2[2] if valid else np.nan
    return u2, v2, f * r3[0] / r3[2], f * r3[1] / r3[2], bool(valid)


def ideal_ray_level(u2, v2, f, alpha):
    """Ideal State 特徵 → H 座標射線（ideal_and_down_from_ray 的反方向，用於跟丟時把推算的 s2 變回視線）。"""
    return camera_rows(-np.pi / 2, -alpha).T @ np.array([u2, v2, f], float)


def project_rows(target, uav, psi, C_H, f):
    """目標世界座標 → 用相機列 C_H 投影的 (u, v, 深度)。"""
    rel = np.asarray(target, float) - np.asarray(uav, float)
    cam = C_H @ (rot_z(psi).T @ rel)
    return f * cam[0] / cam[2], f * cam[1] / cam[2], cam[2]


def los_angles(vec):
    """向量 → (方位角, 仰角)；方位從 +x 逆時針、仰角往下為負。"""
    return float(np.arctan2(vec[1], vec[0])), float(np.arctan2(vec[2], np.hypot(vec[0], vec[1])))


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def ideal_to_down(u2, v2, alpha, f):
    """Eq. 15 / Eq. 54：Ideal State (u2, v2) → 垂直朝下虛擬相機 (u3, v3)。"""
    den = v2 * np.cos(alpha) + f * np.sin(alpha)
    u3 = f * u2 / den
    v3 = f * (v2 * np.sin(alpha) - f * np.cos(alpha)) / den
    return u3, v3


def ideal_depth(v2, alpha, f, H):
    """Ideal State 相機中目標的深度 z_c（平地假設，只用 H 與影像，不做定位）。"""
    return H * f / (v2 * np.cos(alpha) + f * np.sin(alpha))


def image_model(u2, v2, u_psi, V_t, alpha, f, z):
    """Eq. 8：Ideal State 特徵的變化率 (u̇2, v̇2)。numpy 與 CasADi 都能用。"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    du = ((f**2 + u2**2) / f * ca - v2 * sa) * u_psi + f / z * V_t
    dv = (u2 * v2 / f * ca + u2 * sa) * u_psi
    return du, dv


def perception_g(u3, v3, f, R, H):
    """Eq. 20 / 27：g = (u3² + v3²)/f² - R²/H²；R = R_M 是 g1，R = R_M - ΔR 是 g2。"""
    return (u3**2 + v3**2) / f**2 - R**2 / H**2
