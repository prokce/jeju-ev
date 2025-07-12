#!/usr/bin/env python3
"""
Mission-3 Node – 순차적 Waypoint 추종 + Dead-Reckoning
======================================================
• FSM Node 가 퍼블리시하는
    - Point   topic **vehicle_pos**   → 초기 (start_x, start_y)
    - Float32 topic **lidar_yaw**     → 초기 헤딩 ψ₀(°)
  을 수신해 초기화를 수행한다.

• mission_state == 3(Mode C) 에 진입하면, 위 두 토픽이 **모두**
  수신된 이후에 미션이 시작된다. 그 전에는 대기.

• Waypoint 는 Look-ahead 간격으로 등분할 생성하며,
  항상 **현재 인덱스 이후**의 Waypoint 만 추종한다.

+ 추가: LiDAR yaw 수신 시점의 엔코더 카운트를 저장해
        미션 시작 기준점으로 사용한다.
"""

import math
import numpy as np
import rclpy
from rclpy.node        import Node
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg      import Float32, Int32, Int64

# ────────────────────────────────────────────────────────────────
# 상수 정의
ENCODER_PPR      = 1024
ENCODER_EDGES    = ENCODER_PPR * 2
EDGES_PER_WHEEL  = ENCODER_EDGES * (9.0 / 5.0)
WHEEL_DIAMETER   = 0.07                      # m
WHEEL_CIRCUM     = math.pi * WHEEL_DIAMETER
DIST_PER_EDGE    = 1.4 / 21415              # m per encoder edge
DT               = 0.05                     # 20 Hz
# ────────────────────────────────────────────────────────────────


def normalize_angle(a: float) -> float:
    """-π ~ π wrap."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mission3Node(Node):
    def __init__(self):
        super().__init__("mission3_node")

        # ─────────────── 파라미터 ───────────────
        self.declare_parameter("goal_x",      0.0)
        self.declare_parameter("goal_y",      2.5)
        self.declare_parameter("wheelbase",   0.20)
        self.declare_parameter("lookahead",   0.50)
        self.declare_parameter("speed_cmd",   40)
        self.declare_parameter("stop_radius", 0.30)

        # ─────────────── 내부 상태 ───────────────
        self.active        = False           # 미션 On/Off
        self.x = self.y    = 0.0             # 추정 위치
        self.yaw_offset    = None            # IMU – LiDAR yaw
        self.yaw_rad       = 0.0
        self.imu_raw_deg   = 0.0
        self.encoder_count = 0
        self.last_count    = 0
        self.last_t        = self.get_clock().now()

        # 초기 위치/헤딩 수신 여부
        self.init_pos      = None            # (x,y)
        self.init_yaw_deg  = None            # ψ₀
        self.init_ready    = False

        # LiDAR yaw 수신 시점 엔코더 기준점  ←★ 추가
        self.encoder_at_lidar = None

        # 경로 & Waypoint 인덱스
        self.path              = []
        self.current_wp_idx    = 0

        # ─────────────── ROS I/O ───────────────
        # 구독
        self.create_subscription(Point,   "vehicle_pos",   self.cb_vehicle_pos,  10)
        self.create_subscription(Float32, "lidar_yaw",     self.cb_lidar_yaw,    10)
        self.create_subscription(Vector3, "imu/data",      self.cb_imu,          10)
        self.create_subscription(Int64,   "encoder/count", self.cb_enc,          10)
        self.create_subscription(Int32,   "mission_state", self.cb_mstate,       10)

        # 퍼블리시
        self.pub_speed = self.create_publisher(Int32,   "motor_speed",    10)
        self.pub_steer = self.create_publisher(Float32, "steering_angle", 10)
        self.pub_pose  = self.create_publisher(Point,   "vehicle_est_pos",10)

        # 주기 타이머
        self.create_timer(DT, self.timer_cb)

        self.last_mstate = None

    # ─────────────── 콜백 ───────────────
    def cb_vehicle_pos(self, msg: Point):
        self.init_pos = (msg.x, msg.y)
        self._check_init_ready()

    def cb_lidar_yaw(self, msg: Float32):
        self.init_yaw_deg = msg.data

        # ★ LiDAR yaw 수신 순간의 엔코더 카운트를 저장
        if self.encoder_at_lidar is None:
            self.encoder_at_lidar = self.encoder_count
            self.get_logger().debug(f"encoder_at_lidar = {self.encoder_at_lidar}")

        self._check_init_ready()

    def _check_init_ready(self):
        """초기 위치·yaw 모두 수신됐는지 확인."""
        self.init_ready = (self.init_pos is not None and
                           self.init_yaw_deg is not None)

    def cb_imu(self, msg: Vector3):
        self.imu_raw_deg = msg.x

    def cb_enc(self, msg: Int64):
        self.encoder_count = msg.data

    def cb_mstate(self, msg: Int32):
        # ───── 미션 시작 (Mode C 진입) ─────
        if msg.data == 3 and self.last_mstate != 3:
            if not self.init_ready:
                self.get_logger().warn("vehicle_pos / lidar_yaw 수신 전 – 대기 중")
            else:
                self.start_mission()

        # ───── 미션 종료 (Mode C 이탈) ─────
        elif msg.data != 3 and self.last_mstate == 3 and self.active:
            self.stop_mission("미션 종료")

        self.last_mstate = msg.data

    # ─────────────── 미션 제어 ───────────────
    def start_mission(self):
        sx, sy           = self.init_pos
        psi0             = self.init_yaw_deg            # deg
        self.yaw_offset  = self.imu_raw_deg - psi0
        self.x, self.y   = sx, sy

        # ★ 기준점: LiDAR yaw 수신 시점 카운트 → 없으면 현재값
        self.last_count  = (self.encoder_at_lidar
                            if self.encoder_at_lidar is not None
                            else self.encoder_count)

        self.last_t      = self.get_clock().now()
        self.current_wp_idx = 0
        self.active      = True

        # 경로 생성
        gx = self.get_parameter("goal_x").value
        gy = self.get_parameter("goal_y").value
        ld = self.get_parameter("lookahead").value
        total_dist = math.hypot(gx - sx, gy - sy)
        num_pts = max(int(total_dist / ld), 1)
        self.path = [(sx + (gx - sx) * i / num_pts,
                      sy + (gy - sy) * i / num_pts)
                     for i in range(num_pts + 1)]
        self.get_logger().info(f"경로 생성 완료 → {len(self.path)} pts")

    def stop_mission(self, msg: str):
        self.active = False
        self.pub_speed.publish(Int32(data=0))
        self.pub_steer.publish(Float32(data=0.0))
        self.get_logger().info(msg)

    # ─────────────── 주기 타이머 ───────────────
    def timer_cb(self):
        if not self.active:
            return

        # (1) 상태 갱신
        now = self.get_clock().now()
        dt  = (now - self.last_t).nanoseconds * 1e-9
        self.last_t = now if dt > 0 else self.last_t

        self.yaw_rad = math.radians(self.imu_raw_deg - self.yaw_offset)

        dc = self.encoder_count - self.last_count
        self.last_count = self.encoder_count
        ds = dc * DIST_PER_EDGE
        self.x += ds * math.sin(self.yaw_rad)
        self.y += ds * math.cos(self.yaw_rad)

        # (2) 목표 도달 판정
        gx = self.get_parameter("goal_x").value
        gy = self.get_parameter("goal_y").value
        if math.hypot(gx - self.x, gy - self.y) <= self.get_parameter("stop_radius").value:
            self.stop_mission("목표 도달!")
            return

        # (3) Look-ahead Waypoint 선택
        ld = self.get_parameter("lookahead").value
        for idx in range(self.current_wp_idx, len(self.path)):
            px, py = self.path[idx]
            if math.hypot(px - self.x, py - self.y) >= ld:
                self.current_wp_idx = idx
                tx, ty = px, py
                break
        else:
            tx, ty = self.path[-1]
            self.current_wp_idx = len(self.path) - 1

        # (4) Pure Pursuit 조향각 계산
        dx = tx - self.x
        dy = ty - self.y
        alpha = normalize_angle(math.atan2(dx, dy) - self.yaw_rad)
        L     = self.get_parameter("wheelbase").value
        delta = math.atan2(2.0 * L * math.sin(alpha), ld)
        steer_deg = max(min(math.degrees(delta), 20.0), -20.0)

        # (5) 명령 퍼블리시
        self.pub_speed.publish(Int32(data=self.get_parameter("speed_cmd").value))
        self.pub_steer.publish(Float32(data=steer_deg))
        self.pub_pose.publish(Point(x=self.x, y=self.y, z=0.0))

        # (6) 로깅
        self.get_logger().info(
            f"WP {self.current_wp_idx}/{len(self.path)-1} "
            f"→ ({tx:.2f},{ty:.2f}) | "
            f"pos ({self.x:.2f},{self.y:.2f}) | "
            f"steer {steer_deg:.1f}°"
        )


# ────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = Mission3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

