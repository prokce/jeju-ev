#!/usr/bin/env python3
"""
Mission-C Node – LiDAR yaw & 초기 위치(vehicle_pos) 기반
순차 Waypoint 추종 + Dead-Reckoning

· FSM Node (Mode C) 가 퍼블리시하는
    - Point  topic “vehicle_pos”  →  출발 좌표
    - Float32 topic “lidar_yaw”   →  LiDAR 절대 헤딩 ψ₀(°)
  을 받아 초기화한다. 사용자가 따로 start_x, start_y, initial_yaw
  파라미터를 넣을 필요가 없다.

· mission_state == 3(Mode C)에 진입하면, 위 두 토픽이
  모두 수신된 시점에 미션이 시작된다. 그 이전에는 대기.

· IMU yaw – LiDAR yaw = yaw_offset → 주행 중 계속 보정하며
  엔코더 누적거리로 Dead-Reckoning.

ROS 2 Humble · Python 3.10
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float32, Int32, Int64

# ────────────────────────────────────────────────────────────────
# 상수 정의
ENCODER_PPR       = 1024                         # 광학 슬릿 수
ENCODER_EDGES     = ENCODER_PPR * 2              # 2× decoding
EDGES_PER_WHEEL   = ENCODER_EDGES * (9.0 / 5.0)  # 기어비(9:5)
WHEEL_DIAMETER    = 0.07                         # m
WHEEL_CIRCUM      = math.pi * WHEEL_DIAMETER     # m
# 실측(1.4 m 이동 → 21 415 edge)으로 산출한 보정치
DIST_PER_EDGE     = 1.4 / 21415                  # m / edge
DT                = 0.05                         # 20 Hz 주기
# ────────────────────────────────────────────────────────────────

def normalize_angle(a: float) -> float:
    """wrap angle to [-π, π]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class MissionCNode(Node):
    def __init__(self):
        super().__init__("mission_c_node")

        # ───── 파라미터 (목표·차량 계수만 남김) ─────
        self.declare_parameter("goal_x",        -1.5)
        self.declare_parameter("goal_y",        2.0)
        self.declare_parameter("wheelbase",     0.2)
        self.declare_parameter("lookahead",     0.5)
        self.declare_parameter("speed_cmd",     40)
        self.declare_parameter("stop_radius",   0.3)

        # ───── 내부 상태 ─────
        self.active            = False          # 주행 중 여부
        self.request_start     = False          # mission_state==3 수신
        self.x = self.y        = 0.0
        self.yaw_offset        = None           # deg
        self.yaw_rad           = 0.0            # rad
        self.imu_raw_deg       = 0.0
        self.encoder_count     = 0
        self.last_count        = 0
        self.last_t            = self.get_clock().now()

        # LiDAR yaw & vehicle_pos 수신 여부
        self.lidar_yaw_deg     = 0.0
        self.vehicle_pos       = None
        self.has_lidar_yaw     = False
        self.has_vehicle_pos   = False

        # encoder flag
        self.has_encoder = False

        # 경로 & 현재 Waypoint 인덱스
        self.path              = []
        self.current_wp_idx    = 0

        # ───── ROS I/O ─────
        self.create_subscription(Int32,  "mission_state", self.cb_mstate, 10)
        self.create_subscription(Float32,"lidar_yaw",     self.cb_lidar, 10)
        self.create_subscription(Point,  "vehicle_pos",   self.cb_vpos,  10)
        self.create_subscription(Vector3,"imu/data",      self.cb_imu,   10)
        self.create_subscription(Int64,  "encoder/count", self.cb_enc,   10)

        self.pub_speed = self.create_publisher(Int32,  "motor_speed",    10)
        self.pub_steer = self.create_publisher(Float32,"steering_angle", 10)
        self.pub_pose  = self.create_publisher(Point,  "vehicle_est_pos",10)

        self.create_timer(DT, self.timer_cb)

    # ──────────────── 콜백 ────────────────
    def cb_lidar(self, msg: Float32):
        self.lidar_yaw_deg   = msg.data
        self.has_lidar_yaw   = True
        if self.has_encoder:
            self.encoder_at_lidar = self.encoder_count

    def cb_vpos(self, msg: Point):
        self.vehicle_pos     = msg
        self.has_vehicle_pos = True

    def cb_imu(self, msg: Vector3):
        self.imu_raw_deg = msg.x

    def cb_enc(self, msg: Int32):
        self.encoder_count = msg.data
        self.has_encoder = True

    def cb_mstate(self, msg: Int32):
        # Mode C 진입
        if msg.data == 3 and getattr(self, "last_mstate", None) != 3:
            self.request_start = True
            self.get_logger().info("mission_state 3 수신 → 미션 준비")

        # Mode C → 종료
        if msg.data != 3 and self.active:
            self.finish_mission()
        # 상태 추적
        self.last_mstate = msg.data

    # ──────────────── 주 타이머 ────────────────
    def timer_cb(self):
        # 아직 시작요청이 없거나, 필요한 초기 데이터가 준비되지 않음 → 대기
        if not self.active:
            if self.request_start:
                if self.has_lidar_yaw and self.has_vehicle_pos:
                    self.start_mission()
                else:
                    self.get_logger().debug("LiDAR yaw / vehicle_pos 대기 중…")
            return

        # 1) Dead-Reckoning으로 현 위치 업데이트
        now = self.get_clock().now()
        dt  = (now - self.last_t).nanoseconds * 1e-9
        self.last_t = now if dt > 0 else self.last_t

        # 실시간 IMU yaw 보정
        self.yaw_rad = math.radians(self.imu_raw_deg - self.yaw_offset)

        # 엔코더 길이 적분
        dc = self.encoder_count - self.last_count
        self.last_count = self.encoder_count
        ds = dc * DIST_PER_EDGE
        self.x += ds * math.sin(self.yaw_rad)
        self.y += ds * math.cos(self.yaw_rad)

        # 2) 목표 도착 검사
        gx = self.get_parameter("goal_x").value
        gy = self.get_parameter("goal_y").value
        stop_r = self.get_parameter("stop_radius").value
        if math.hypot(gx - self.x, gy - self.y) <= stop_r:
            self.finish_mission()
            self.active = False
            self.get_logger().info("목표 도달!")
            return

        # 3) 다음 Waypoint 선택 (현재 인덱스 이후만)
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

        # 4) Pure-Pursuit 조향각
        dx = tx - self.x
        dy = ty - self.y
        alpha = normalize_angle(math.atan2(dx, dy) - self.yaw_rad)
        L     = self.get_parameter("wheelbase").value
        delta = math.atan2(2.0 * L * math.sin(alpha), ld)
        steer_deg = math.degrees(delta)
        steer_deg = max(min(steer_deg, 20.0), -20.0)

        # 5) 토픽 퍼블리시
        self.pub_speed.publish(Int32(data=self.get_parameter("speed_cmd").value))
        self.pub_steer.publish(Float32(data=steer_deg))
        self.pub_pose.publish(Point(x=self.x, y=self.y, z=0.0))

        # 디버그 로그
        self.get_logger().info(
            f"[WP {self.current_wp_idx}/{len(self.path)-1}] "
            f"목표({tx:.2f},{ty:.2f}) | "
            f"pos({self.x:.2f},{self.y:.2f}) | "
            f"yaw {math.degrees(self.yaw_rad):.1f}° | steer {steer_deg:.1f}°"
        )

    # ──────────────── 미션 시작/종료 헬퍼 ────────────────
    def start_mission(self):
        # LiDAR yaw & vehicle_pos 기반 초기화
        self.yaw_offset = self.imu_raw_deg - self.lidar_yaw_deg
        self.x, self.y  = self.vehicle_pos.x, self.vehicle_pos.y
        self.last_count = getattr(self, "encoder_at_lidar", self.encoder_count)
        self.last_t     = self.get_clock().now()
        self.current_wp_idx = 0

        # 경로 생성 (등간격 Waypoints)
        gx = self.get_parameter("goal_x").value
        gy = self.get_parameter("goal_y").value
        sx, sy = self.x, self.y
        total_dist = math.hypot(gx - sx, gy - sy)
        ld = self.get_parameter("lookahead").value
        num_pts = max(int(total_dist / ld), 1)
        self.path = [
            (sx + (gx - sx) * i / num_pts,
             sy + (gy - sy) * i / num_pts)
            for i in range(num_pts + 1)
        ]

        self.active = True
        self.request_start = False
        self.get_logger().info(
            f"미션 시작! ψ₀={self.lidar_yaw_deg:.2f}° | "
            f"start({self.x:.2f},{self.y:.2f}) → goal({gx},{gy}), "
            f"{len(self.path)} WPs"
        )

    def finish_mission(self):
        self.pub_speed.publish(Int32(data=0))
        self.pub_steer.publish(Float32(data=0.0))
        self.active = False
        self.request_start = False
        self.get_logger().info("미션 종료")
    # ───────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = MissionCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

