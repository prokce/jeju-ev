#!/usr/bin/env python3
"""
Mission-3 Node – 경유지(0,0.5) → 최종 목표 2-스테이지 Waypoint 추종
==================================================================
1) Mode C 진입 시:
      (start_x,start_y) → (0,0.5)  한 번만 Path 생성
2) (0,0.5) 중심 0.30 m 이내 도달하면:
      Path = (현위치) → (goal_x,goal_y) 로 즉시 갱신
3) Pure-Pursuit 로 항상 현재 인덱스 이후 Waypoint 추종
"""

import math, rclpy
from rclpy.node        import Node
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg      import Float32, Int32, Int64

# ────────────────────────────────────────────────────────────────
ENCODER_PPR      = 1024
ENCODER_EDGES    = ENCODER_PPR * 2
DIST_PER_EDGE    = 1.4 / 21415        # m/edge  (실측값)
DT               = 0.05               # 20 Hz
INT_X, INT_Y     = 0.0, 0.5           # 첫 경유지
# ────────────────────────────────────────────────────────────────


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mission3Node(Node):
    def __init__(self):
        super().__init__("mission3_node")

        # ────── 파라미터 ──────
        self.declare_parameter("goal_x",      -3.0)
        self.declare_parameter("goal_y",      2.0)
        self.declare_parameter("wheelbase",   0.20)
        self.declare_parameter("lookahead",   0.50)
        self.declare_parameter("speed_cmd",   40)
        self.declare_parameter("stop_radius", 0.30)
        self.declare_parameter("intermediate_radius", 0.30)

        # ────── 내부 상태 ──────
        self.active              = False
        self.x = self.y          = 0.0
        self.yaw_offset          = None
        self.yaw_rad             = 0.0
        self.imu_raw_deg         = 0.0
        self.encoder_count       = 0
        self.last_count          = 0
        self.last_t              = self.get_clock().now()

        self.init_pos            = None
        self.init_yaw_deg        = None
        self.init_ready          = False
        self.encoder_at_lidar    = None

        self.path                = []
        self.current_wp_idx      = 0
        self.intermediate_reached = False

        # ────── ROS I/O ──────
        self.create_subscription(Point,   "vehicle_pos",   self.cb_vehicle_pos,  10)
        self.create_subscription(Float32, "lidar_yaw",     self.cb_lidar_yaw,    10)
        self.create_subscription(Vector3, "imu/data",      self.cb_imu,          10)
        self.create_subscription(Int64,   "encoder/count", self.cb_enc,          10)
        self.create_subscription(Int32,   "mission_state", self.cb_mstate,       10)

        self.pub_speed = self.create_publisher(Int32,   "motor_speed",    10)
        self.pub_steer = self.create_publisher(Float32, "steering_angle", 10)
        self.pub_pose  = self.create_publisher(Point,   "vehicle_est_pos",10)

        self.create_timer(DT, self.timer_cb)
        self.last_mstate = None

    # ─────────────── 콜백 ───────────────
    def cb_vehicle_pos(self, msg: Point):
        self.init_pos = (msg.x, msg.y)
        self._check_init_ready()

    def cb_lidar_yaw(self, msg: Float32):
        self.init_yaw_deg = msg.data
        if self.encoder_at_lidar is None:
            self.encoder_at_lidar = self.encoder_count
        self._check_init_ready()

    def _check_init_ready(self):
        self.init_ready = (self.init_pos is not None and
                           self.init_yaw_deg is not None)

    def cb_imu(self, msg: Vector3):
        self.imu_raw_deg = msg.x

    def cb_enc(self, msg: Int64):
        self.encoder_count = msg.data

    def cb_mstate(self, msg: Int32):
        if msg.data == 3 and self.last_mstate != 3:
            if self.init_ready:
                self.start_mission()
            else:
                self.get_logger().warn("vehicle_pos / lidar_yaw 수신 전 – 대기중")
        elif msg.data != 3 and self.last_mstate == 3 and self.active:
            self.stop_mission("미션 종료")
        self.last_mstate = msg.data

    # ─────────────── 미션 제어 ───────────────
    def start_mission(self):
        sx, sy           = self.init_pos
        psi0             = self.init_yaw_deg
        self.yaw_offset  = self.imu_raw_deg - psi0
        self.x, self.y   = sx, sy
        self.last_count  = (self.encoder_at_lidar
                            if self.encoder_at_lidar is not None
                            else self.encoder_count)
        self.last_t      = self.get_clock().now()
        self.current_wp_idx = 0
        self.intermediate_reached = False
        self.active      = True

        self.path = self._make_segment(sx, sy, INT_X, INT_Y) + [(INT_X, INT_Y)]
        self.get_logger().info(f"Path-1 생성 → {len(self.path)} pts")

    def _make_segment(self, ax, ay, bx, by):
        ld   = self.get_parameter("lookahead").value
        dist = math.hypot(bx - ax, by - ay)
        n    = max(int(dist / ld), 1)
        return [(ax + (bx - ax) * i / n,
                 ay + (by - ay) * i / n) for i in range(n)]

    def _switch_to_goal_path(self):
        gx = self.get_parameter("goal_x").value
        gy = self.get_parameter("goal_y").value
        self.path = self._make_segment(self.x, self.y, gx, gy) + [(gx, gy)]
        self.current_wp_idx = 0
        self.intermediate_reached = True
        self.get_logger().info(f"Path-2(→Goal) 재생성 → {len(self.path)} pts")

    def stop_mission(self, msg):
        self.active = False
        self.pub_speed.publish(Int32(data=0))
        self.pub_steer.publish(Float32(data=0.0))
        self.get_logger().info(msg)

    # ─────────────── 주기 타이머 ───────────────
    def timer_cb(self):
        if not self.active:
            return

        # --- Dead-Reckoning ---
        now = self.get_clock().now()
        dt  = (now - self.last_t).nanoseconds * 1e-9
        if dt <= 0:
            return
        self.last_t = now

        self.yaw_rad = math.radians(self.imu_raw_deg - self.yaw_offset)
        dc = self.encoder_count - self.last_count
        self.last_count = self.encoder_count
        ds = dc * DIST_PER_EDGE
        self.x += ds * math.sin(self.yaw_rad)
        self.y += ds * math.cos(self.yaw_rad)

        # --- 중간 경유지 판정 ---
        if (not self.intermediate_reached and
            math.hypot(self.x - INT_X, self.y - INT_Y)
                <= self.get_parameter("intermediate_radius").value):
            self._switch_to_goal_path()

        # --- 최종 목표 도달 판정 ---
        gx = self.get_parameter("goal_x").value
        gy = self.get_parameter("goal_y").value
        if math.hypot(gx - self.x, gy - self.y) <= self.get_parameter("stop_radius").value:
            self.stop_mission("목표 도달!")
            return

        # --- Look-ahead Waypoint 선택 ---
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

        # --- Pure-Pursuit 조향각 ---
        alpha = normalize_angle(math.atan2(tx - self.x, ty - self.y) - self.yaw_rad)
        L     = self.get_parameter("wheelbase").value
        delta = math.atan2(2.0 * L * math.sin(alpha), ld)
        steer_deg = max(min(math.degrees(delta), 20.0), -20.0)

        # --- 퍼블리시 ---
        self.pub_speed.publish(Int32(data=self.get_parameter("speed_cmd").value))
        self.pub_steer.publish(Float32(data=steer_deg))
        self.pub_pose.publish(Point(x=self.x, y=self.y, z=0.0))

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

