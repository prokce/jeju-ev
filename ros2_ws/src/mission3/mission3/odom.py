#!/usr/bin/env python3
"""
Mission-3 Node – 순차적 Waypoint 추종 + Dead-Reckoning
- (NEW) 중간 경유지: (0.0, 0.5) 중심, 반경 0.5 m 진입 후 최종 목적지
- 경로 생성 시 Waypoint 인덱스 추적
- 항상 현재 인덱스 이후의 Waypoint만 선택
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float32, Int32, Int64

# ────────────────────────────────────────────────────────────────
# 상수 정의
ENCODER_PPR     = 1024
ENCODER_EDGES   = ENCODER_PPR * 2
EDGES_PER_WHEEL = ENCODER_EDGES * (9.0 / 5.0)
WHEEL_DIAMETER  = 0.07          # m
WHEEL_CIRCUM    = math.pi * WHEEL_DIAMETER
DIST_PER_EDGE   = 1.4 / 21415   # m per encoder edge
DT              = 0.05          # 20 Hz

# (NEW) 중간 경유지 파라미터
MID_X       = 0.0
MID_Y       = 0.0
MID_RADIUS  = 0.3               # m
# ────────────────────────────────────────────────────────────────


def normalize_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mission3Node(Node):
    def __init__(self):
        super().__init__("mission3_node")

        # 파라미터
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", -2.85)
        self.declare_parameter("initial_yaw", 0.0)
        self.declare_parameter("goal_x", -3.0)
        self.declare_parameter("goal_y", 2.0)
        self.declare_parameter("wheelbase", 0.2)
        self.declare_parameter("lookahead", 0.5)
        self.declare_parameter("speed_cmd", 40)
        self.declare_parameter("stop_radius", 0.3)
        self.last_mstate = None

        # 내부 상태
        self.active          = False
        self.x = self.y      = 0.0
        self.yaw_offset      = None  # deg
        self.yaw_rad         = 0.0
        self.imu_raw_deg     = 0.0
        self.encoder_count   = 0
        self.last_count      = 0
        self.last_t          = self.get_clock().now()

        # (NEW) 경유지/목적지 관리
        self.targets = []             # [(x, y), …]
        self.target_idx = 0

        # 경로 및 Waypoint 인덱스
        self.path            = []
        self.current_wp_idx  = 0

        # 구독/퍼블리시
        self.create_subscription(Int32,  "mission_state",  self.cb_mstate, 10)
        self.create_subscription(Vector3,"imu/data",       self.cb_imu,    10)
        self.create_subscription(Int64,  "encoder/count",  self.cb_enc,    10)
        self.pub_speed = self.create_publisher(Int32,   "motor_speed",     10)
        self.pub_steer = self.create_publisher(Float32, "steering_angle",  10)
        self.pub_pose  = self.create_publisher(Point,   "vehicle_est_pos", 10)

        self.create_timer(DT, self.timer_cb)

    # ────────────── 콜백 ──────────────
    def cb_imu(self, msg: Vector3):
        self.imu_raw_deg = msg.x

    def cb_enc(self, msg: Int32):
        self.encoder_count = msg.data

    def cb_mstate(self, msg: Int32):
        # ===== 미션 시작 =====
        if msg.data == 3 and self.last_mstate != 3:
            sx = self.get_parameter("start_x").value
            sy = self.get_parameter("start_y").value
            gx = self.get_parameter("goal_x").value
            gy = self.get_parameter("goal_y").value
            psi0 = self.get_parameter("initial_yaw").value

            self.yaw_offset = self.imu_raw_deg - psi0
            self.x, self.y = sx, sy
            self.last_count = self.encoder_count
            self.last_t = self.get_clock().now()
            self.active = True
            self.current_wp_idx = 0

            # (NEW) 목표 리스트 설정
            self.targets = [(MID_X, MID_Y), (gx, gy)]
            # 이미 중간 반경 안이라면 첫 타깃 건너뜀
            if math.hypot(self.x - MID_X, self.y - MID_Y) <= MID_RADIUS:
                self.target_idx = 1
            else:
                self.target_idx = 0

            # 초기 경로 생성
            self._generate_path()
            self.get_logger().info(
                f"미션 시작 → 경유지 {self.targets[0]}, 최종 {self.targets[1]}"
            )

        # ===== 미션 종료 =====
        elif msg.data != 3 and self.last_mstate == 3 and self.active:
            self._mission_finish("외부 명령으로 종료")
        self.last_mstate = msg.data

    # ────────────── 내부 함수 ──────────────
    def _generate_path(self):
        """현재 위치(self.x, self.y)에서 현 타깃까지 등간격 Waypoints 생성."""
        tx, ty = self.targets[self.target_idx]
        sx, sy = self.x, self.y
        total_dist = math.hypot(tx - sx, ty - sy)
        ld = self.get_parameter("lookahead").value
        num_pts = max(int(total_dist / ld), 1)
        self.path = [
            (sx + (tx - sx) * i / num_pts,
             sy + (ty - sy) * i / num_pts)
            for i in range(num_pts + 1)
        ]
        self.current_wp_idx = 0
        self.get_logger().info(
            f"경로 재생성 → 타깃 {self.target_idx+1}/{len(self.targets)} "
            f"({tx:.2f},{ty:.2f})  |  Waypoints {len(self.path)}"
        )

    def _mission_finish(self, msg: str):
        self.active = False
        self.pub_speed.publish(Int32(data=0))
        self.pub_steer.publish(Float32(data=0.0))
        self.get_logger().info(f"미션 종료: {msg}")

    # ────────────── 타이머 ──────────────
    def timer_cb(self):
        if not self.active:
            return

        # 1) 상태 갱신
        now = self.get_clock().now()
        dt = (now - self.last_t).nanoseconds * 1e-9
        self.last_t = now if dt > 0 else self.last_t
        self.yaw_rad = math.radians(self.imu_raw_deg - self.yaw_offset)

        # 엔코더 → 거리
        dc = self.encoder_count - self.last_count
        self.last_count = self.encoder_count
        ds = dc * DIST_PER_EDGE
        self.x += ds * math.sin(self.yaw_rad)
        self.y += ds * math.cos(self.yaw_rad)

        # 2) 타깃 도착 판정
        tx_goal, ty_goal = self.targets[self.target_idx]
        tgt_radius = MID_RADIUS if self.target_idx == 0 else self.get_parameter("stop_radius").value
        if math.hypot(tx_goal - self.x, ty_goal - self.y) <= tgt_radius:
            if self.target_idx < len(self.targets) - 1:
                # (NEW) 다음 타깃으로 전환
                self.target_idx += 1
                self._generate_path()
            else:
                # 최종 목적지 도달
                self._mission_finish("최종 목적지 도달!")
                return

        # 3) 순차 Waypoint 선택
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

        # 4) Pure Pursuit 제어
        dx = tx - self.x
        dy = ty - self.y
        alpha = normalize_angle(math.atan2(dx, dy) - self.yaw_rad)
        L = self.get_parameter("wheelbase").value
        delta = math.atan2(2.0 * L * math.sin(alpha), ld)
        steer_deg = max(min(math.degrees(delta), 20.0), -20.0)

        # 5) 명령 발행
        self.pub_speed.publish(
            Int32(data=self.get_parameter("speed_cmd").value)
        )
        self.pub_steer.publish(Float32(data=steer_deg))
        self.pub_pose.publish(Point(x=self.x, y=self.y, z=0.0))

        # 6) 로깅
        self.get_logger().info(
            f"[타깃 {self.target_idx+1}/{len(self.targets)}] "
            f"WP {self.current_wp_idx}/{len(self.path)}  | "
            f"목표 ({tx:.2f},{ty:.2f})  | "
            f"위치 ({self.x:.2f},{self.y:.2f})"
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

