#!/usr/bin/env python3
"""
FSM Node (RPLIDAR A2) – 벽 중점 기준 로컬 프레임 생성
===================================================
• ModeA : LiDAR ±0.45π 거리 ≤ 2.5 m → ModeB
• ModeB : (시간 조건) 5 초 경과 → motor_speed 0.0 퍼블리시,
          1 초 후 ModeC 전환
• ModeC : ±90° (A)·B점으로 α 계산 → yaw 퍼블리시
          & 차량 위치(벽 중점 원점) 퍼블리시
좌표계
-------
 origin = 두 벽 끝점 중점
 +X     = origin→오른쪽 벽 끝
 –X     = origin→왼쪽 벽 끝
 +Y     = 벽에서 밖(탈출) 방향 → 차량은 y<0 에서 출발
"""
import math, numpy as np, rclpy
from rclpy.node          import Node
from sensor_msgs.msg     import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg   import Point
from std_msgs.msg        import String, Int32, Float32, Header, ColorRGBA

# ────────────────────────────────────────────────────────────────
class FSM:
    def __init__(self):
        self.state = "ModeA"
        self.left_marker  = None
        self.right_marker = None

    def update(self, l_left, l_right,
               left_cnt, right_cnt,
               l_70, l_m70):
        """ModeA → ModeB 만 판단. ModeB → ModeC 전환은 FSMNode가 시간 기반으로 처리."""
        if self.state == "ModeA":
            if math.isfinite(l_left) and math.isfinite(l_right):
                if l_left <= 1.5 and l_right <= 1.5:
                    self.state = "ModeB"
        # ModeB 에서는 유지, 다른 전환 조건 없음

# ────────────────────────────────────────────────────────────────
class FSMNode(Node):
    def __init__(self):
        super().__init__("fsm_node")
        self.fsm = FSM()

        # ROS I/O
        self.create_subscription(LaserScan, "/scan",             self.cb_scan,  10)
        self.create_subscription(Marker,    "left_wall_points",  self.cb_left,  10)
        self.create_subscription(Marker,    "right_wall_points", self.cb_right, 10)

        self.pub_fsm   = self.create_publisher(String,  "fsm_state",     10)
        self.pub_msn   = self.create_publisher(Int32,   "mission_state", 10)
        self.pub_veh   = self.create_publisher(Point,   "vehicle_pos",   10)
        self.pub_yaw   = self.create_publisher(Float32, "lidar_yaw",     10)
        self.pub_beams = self.create_publisher(Marker,  "beam_markers",  10)
        self.pub_speed = self.create_publisher(Int32, "motor_speed",   10)  # ★ 추가

        self.create_timer(0.10, self.timer_cb)   # 10 Hz

        # internals
        self.scan_ranges = []
        self.angle_min = self.angle_inc = 0.0
        self.l45 = self.r45 = float('inf')
        self.l70 = self.m70 = float('inf')
        self.left_marker = self.right_marker = None
        self.yaw_sent = False
        self.pos_sent = False

        # ModeB 타이밍 관리용 ★
        self.modeb_enter_time = None
        self.motor_zero_sent  = False

    # ─────────────── LaserScan 콜백 ───────────────
    def cb_scan(self, msg: LaserScan):
        self.scan_ranges = np.asarray(msg.ranges)
        self.angle_min   = msg.angle_min
        self.angle_inc   = msg.angle_increment

        def rng(rad):
            i = int(round((rad - self.angle_min) / self.angle_inc))
            return msg.ranges[i] if 0 <= i < len(msg.ranges) else float('inf')

        self.l45  = rng(+0.5*math.pi)
        self.r45  = rng(-0.5*math.pi)
        self.l70  = rng( math.radians( 70))
        self.m70  = rng( math.radians(-70))

    def cb_left(self, msg):  self.left_marker  = msg; self.fsm.left_marker  = msg
    def cb_right(self, msg): self.right_marker = msg; self.fsm.right_marker = msg

    # ─────────────── 주기 타이머 ───────────────
    def timer_cb(self):
        prev_state = self.fsm.state

        # ModeA → ModeB 판정
        self.fsm.update(self.l45, self.r45,
                        len(self.left_marker.points)  if self.left_marker  else 0,
                        len(self.right_marker.points) if self.right_marker else 0,
                        self.l70, self.m70)

        # ★ ModeB 시간 기반 처리 ★
        if self.fsm.state == "ModeB":
            if self.modeb_enter_time is None:          # ModeB 첫 진입
                self.modeb_enter_time = self.get_clock().now()
                self.motor_zero_sent  = False
            else:
                elapsed = (self.get_clock().now() - self.modeb_enter_time).nanoseconds * 1e-9
                if elapsed >= 8.0 and not self.motor_zero_sent:
                    self.pub_speed.publish(Int32(data=0))  # motor_speed 0.0 퍼블리시
                    self.motor_zero_sent = True
                    self.get_logger().info("ModeB 5 초 경과 → motor_speed 0.0 퍼블리시")
                if elapsed >= 8.5:                     # 1 초 더 기다린 후 ModeC 전환
                    self.fsm.state = "ModeC"
        else:
            # ModeB 를 벗어나면 타이머 초기화
            self.modeb_enter_time = None
            self.motor_zero_sent  = False

        curr_state = self.fsm.state

        # 상태 퍼블리시
        self.pub_fsm.publish(String(data=curr_state))
        self.pub_msn.publish(Int32(data={"ModeA":1,"ModeB":2,"ModeC":3}[curr_state]))

        # ModeB → ModeC 진입 직후: LiDAR yaw 계산
        if prev_state == "ModeB" and curr_state == "ModeC" and not self.yaw_sent:
            self.compute_and_publish_yaw()

        # ModeC: 차량 좌표 한 번 퍼블리시
        if curr_state == "ModeC" and self.left_marker and self.right_marker and not self.pos_sent:
            self.publish_vehicle_pos()

        # RViz 70°/-70° 빔 시각화
        self.publish_beam_markers()

    # ─────────────── 70°/-70° 빔 표시 ───────────────
    def publish_beam_markers(self):
        m = Marker()
        m.header = Header(frame_id="laser", stamp=self.get_clock().now().to_msg())
        m.ns, m.id, m.type, m.action = "beams", 0, Marker.LINE_LIST, Marker.ADD
        m.scale.x = 0.03
        m.color   = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        o = Point(x=0.0,y=0.0,z=0.0)
        for a in ( math.radians(80), math.radians(-80) ):
            m.points += [o, Point(x=math.cos(a), y=math.sin(a), z=0.0)]
        self.pub_beams.publish(m)

    # ─────────────── LiDAR yaw 계산 ───────────────
    def compute_and_publish_yaw(self):
        cand = []
        for name, ang in (("right", math.radians(90)),
                          ("left",  math.radians(-90))):
            i = int(round((ang - self.angle_min)/self.angle_inc))
            if 0 <= i < len(self.scan_ranges) and math.isfinite(self.scan_ranges[i]):
                r = self.scan_ranges[i]
                cand.append((name, r, r*math.cos(ang), r*math.sin(ang)))
        if not cand:
            self.get_logger().warn("LiDAR yaw 계산 실패 (A점 없음)")
            return
        name_a, r_a, ax, ay = min(cand, key=lambda t: t[1])

        n = len(self.scan_ranges)
        angs = self.angle_min + np.arange(n)*self.angle_inc
        deg  = (angs*180/math.pi) % 360
        mask = ((deg>=170)&(deg<=300)) if name_a=="left" else ((deg>=60)&(deg<=190))
        valid = mask & np.isfinite(self.scan_ranges)
        if not np.any(valid):
            self.get_logger().warn("LiDAR yaw 계산 실패 (B점 없음)")
            return
        i_b   = np.argmin(np.where(valid, self.scan_ranges, np.inf))
        r_b   = self.scan_ranges[i_b]
        bx,by = r_b*math.cos(angs[i_b]), r_b*math.sin(angs[i_b])

        dot = ax*bx + ay*by
        alpha = math.degrees( math.acos( max(min(dot/(r_a*r_b),1), -1) ) )
        sign  = 1 if (name_a=="right" and ax>bx) or (name_a=="left" and ax<bx) else -1
        yaw_lidar = sign * alpha

        self.pub_yaw.publish(Float32(data=yaw_lidar))
        self.yaw_sent = True
        self.get_logger().info(f"LiDAR yaw = {yaw_lidar:.2f}° (A={name_a})")

    # ─────────────── 차량 위치 퍼블리시 ───────────────
    def publish_vehicle_pos(self):
        L = max(self.left_marker.points,  key=lambda p: math.hypot(p.x,p.y))
        R = max(self.right_marker.points, key=lambda p: math.hypot(p.x,p.y))

        mid_x, mid_y = (L.x+R.x)/2, (L.y+R.y)/2
        dx, dy       =  R.x-mid_x,  R.y-mid_y
        theta        = math.atan2(dy, dx)

        tx, ty = -mid_x, -mid_y
        ct, st = math.cos(-theta), math.sin(-theta)
        x0 =  tx*ct - ty*st
        y0 =  tx*st + ty*ct
        if y0 > 0:
            y0 = -y0

        self.pub_veh.publish(Point(x=x0, y=y0, z=0.0))
        self.pos_sent = True
        self.get_logger().info(f"vehicle_pos → ({x0:.2f}, {y0:.2f})")

# ────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = FSMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

