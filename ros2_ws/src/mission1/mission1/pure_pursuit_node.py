import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Int32
import math
import numpy as np

from scipy.ndimage import gaussian_filter1d

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

curvature_buffer = []
last_angle = 0.0
angle_buffer = []
previous_pwm = 40.0  # 초기 속도는 직선 주행 가정

# ======================================================================
#                   속도와 조향각 계산을 위한 전처리 관련 함수
# ======================================================================

### === -pi ~ +pi 범위로 각도 정규화 ===
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi    
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

### === path 기반 pure pursuit 사용하여 조향 각도 계산 ===
def get_steering_pure_pursuit(path, L=0.2, Ld=2.0):
    global last_angle
    if not path:
        return last_angle
    tgt = next(((x, y) for x, y in path if math.hypot(x, y) >= Ld), path[-1])
    alpha = math.atan2(tgt[1], tgt[0])
    delta = math.atan2(2 * L * math.sin(alpha), Ld)
    ang = math.degrees(delta)   # 수학적으로는 좌회전 양수
    last_angle = -ang           # 실제 차량 조향 시스템에 맞춰 좌회전을 음수로 변환
    return -ang          

### === 곡률 계산 함수 ===
def estimate_curvature(path):
    if path is None or len(path) < 3:
        return 0.0
    xs, ys = zip(*path)
    dx = np.diff(xs)
    dy = np.diff(ys)
    d2x = np.diff(dx)
    d2y = np.diff(dy)
    denom = (dx[:-1]**2 + dy[:-1]**2)**1.5
    denom = np.where(denom == 0, 1e-6, denom)  # ⚠️ 0으로 나누는 것 방지
    kappa = np.abs(d2x * dy[:-1] - d2y * dx[:-1]) / denom
    return float(np.median(kappa))

### === 곡률 기반 Ld 조정 함수 ===
def get_dynamic_ld_from_curvature(curv, min_ld=0.8, max_ld=2.0, threshold=0.67):
    """
    곡률 기반 Ld 조절 함수
    - curv = 0        → Ld = max_ld
    - curv = threshold → Ld = min_ld
    - curv > threshold → Ld = min_ld 고정
    - 그 사이 값은 선형 보간
    """
    curv = max(0.0, curv)

    if curv >= threshold:
        return min_ld
    else:
        ratio = 1.0 - (curv / threshold)
        return min_ld + (max_ld - min_ld) * ratio
    

# 전역 변수
previous_ld = 2.3  # 초기 Ld 값 (중간값)

def get_smoothed_ld(target_ld, recovery_step=0.1):
    global previous_ld
    if target_ld < previous_ld:
        # 커브가 심해졌다면 → 즉시 감소
        previous_ld = target_ld
    elif target_ld > previous_ld + recovery_step:
        # 완만해졌다면 → 천천히 증가
        previous_ld += recovery_step
    else:
        # 변화폭 작으면 바로 적용
        previous_ld = target_ld
    return previous_ld



# =====================================================================
#                           조향각 관련 함수
# =====================================================================

### === 차량의 헤딩과 경로 각도의 차이 기반 조향 보정 적용 ===
'''
def apply_heading_correction(steer, path):
    if not path or len(path) < 2:
        return steer, 0.0

    lookahead_dist = 1.0
    for i in range(1, len(path)):
        x, y = path[i]
        if math.hypot(x, y) >= lookahead_dist:
            target_x, target_y = x, y
            break
    else:
        target_x, target_y = path[-1]

    path_theta = math.atan2(target_y, target_x)
    vehicle_theta = 0.0
    heading_error = normalize_angle(path_theta - vehicle_theta)

    gain = 5.0  # 👉 조정 가능한 고정값
    correction_deg = math.degrees(heading_error) * gain

    return steer + correction_deg, heading_error
'''

### === 조향각 계산 보조 함수 ===
def clamp_angle(a, max_angle=20):
    return float(np.clip(a, -max_angle, max_angle))

def limit_angle_change(a, prev, max_delta=0.5):
    d = a - prev
    return prev + np.clip(d, -max_delta, max_delta) if abs(d) > max_delta else a

def smooth_angle(v, window=13):
    angle_buffer.append(v)
    if len(angle_buffer) > window:
        angle_buffer.pop(0)
    return float(np.mean(angle_buffer))

# ===================================================================
#                           속도 관련 함수
# ===================================================================

### === 조향각 기반 속도 제어 함수 ===
def get_speed_from_steering_angle(angle_deg, min_pwm=30, max_pwm=70):
    """
    조향각 기반 PWM 속도 설정:
    - 0~10도: 70 → 40 선형 감소
    - 10~20도: 40 → 30 선형 감소
    - 20도 초과: 30 고정
    """
    abs_angle = abs(angle_deg)

    if abs_angle <= 10.0:
        # 구간 1: 0~10도 → 70~40 선형 감소
        ratio = abs_angle / 10.0
        return max_pwm - (max_pwm - 40) * ratio
    elif abs_angle <= 20.0:
        # 구간 2: 10~20도 → 40~30 선형 감소
        ratio = (abs_angle - 10.0) / 10.0
        return 40 - (10 * ratio)
    else:
        # 구간 3: 20도 초과 → 최소 속도 고정
        return min_pwm
    

def get_smoothed_speed(target_pwm, smoothing_step=0.1):
    global previous_pwm
    if target_pwm < previous_pwm:
        # 속도 줄일 때는 즉시 감소 허용
        previous_pwm = target_pwm
    elif target_pwm > previous_pwm + smoothing_step:
        # 속도 증가는 smoothing_step만큼 제한
        previous_pwm += smoothing_step
    else:
        # smoothing_step보다 작으면 바로 증가
        previous_pwm = target_pwm
    return previous_pwm













class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.subscription = self.create_subscription(Path, '/lane/path', self.path_callback, 10)
        self.steer_pub = self.create_publisher(Float32, '/steering_angle', 10)
        self.speed_pub = self.create_publisher(Int32, '/motor_speed', 10)
        self.bridge = CvBridge()
        

    def path_callback(self, msg):
        global last_angle

        path_points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        if not path_points:
            self.get_logger().warn("Path is empty")
            stop_speed = Int32()
            stop_speed.data = 32
            self.speed_pub.publish(stop_speed)
            stop_steer = Float32()
            stop_steer.data = 0.0
            self.steer_pub.publish(stop_steer)
            last_angle = 0.0
            return
    


        frame_id = msg.header.frame_id.lower()
        if frame_id in ["left_only", "right_only"] or len(path_points) < 5:
            curv = 0.0
            target_ld = 0.8
            dynamic_ld = 0.8
            min_pwm = 30  # 한쪽 차선만 검출 시, 안전한 최소 속도 보장
        else:
            curv = estimate_curvature(path_points)
            target_ld = get_dynamic_ld_from_curvature(curv)
            dynamic_ld = get_smoothed_ld(target_ld, recovery_step=0.05)
            min_pwm = 30  # 양쪽 차선 있을 때는 더 낮게 허용 가능




        raw_steer = get_steering_pure_pursuit(path_points, L=0.2, Ld=dynamic_ld)

        # corrected_steer, _ = apply_heading_correction(raw_steer, path_points)
        # target_pwm = get_speed_from_steering_angle(corrected_steer)
        target_pwm = get_speed_from_steering_angle(raw_steer, min_pwm=min_pwm)
        smoothed_pwm = get_smoothed_speed(target_pwm)
        # self.vehicle_speed = smoothed_pwm
        self.vehicle_speed = 35

        smoothed = smooth_angle(raw_steer)
        limited = limit_angle_change(smoothed, last_angle)
        final_steer = clamp_angle(limited)
        last_angle = final_steer


        # 조향각 퍼블리시
        steer_msg = Float32()
        steer_msg.data = final_steer
        self.steer_pub.publish(steer_msg)

        # 속도 퍼블리시
        speed_msg = Int32()
        # speed_msg.data = int(self.vehicle_speed)
        speed_msg.data = 35
        self.speed_pub.publish(speed_msg)


        # RawSteer, FinalSteer: 우회전 +, 좌회전 -
        self.get_logger().info(
        f"[Curv] {curv:.2f}, [TargetLd] {target_ld:.2f}, [SmoothedLd] {dynamic_ld:.2f} "
        f"[RawSteer] {raw_steer:.2f}°, [FinalSteer] {final_steer:.2f}° "
        f"[TargetPWM] {target_pwm:.1f}, [SmoothedPWM] {smoothed_pwm:.1f}"
        )

        self.latest_path = path_points
        self.latest_dynamic_ld = dynamic_ld

    

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(PurePursuitNode())
    rclpy.shutdown()
