#!/usr/bin/env python3
"""
Wall & Path Nodes (angle/speed gated only)
========================================
세 개의 노드를 한 프로세스에서 실행합니다.

1. **RightWallBSplinePointsPublisher**
   – 라이다 스캔으로부터 오른쪽 벽 포인트를 B-스플라인 보간 → Marker(POINTS)
2. **LeftWallBSplinePointsPublisher**
   – 라이다 스캔으로부터 왼쪽 벽 포인트를 B-스플라인 보간 → Marker(POINTS)
3. **FinalPathPublisher**
   – 양쪽 벽의 중간선을 따라 최종 경로를 생성 후 Pure-Pursuit
   – `steering_angle`, `motor_speed` **퍼블리시만** `mission_state == 2` 일 때 수행
4. **Rear-Axle Marker (NEW)**
   – 차량 후륜축 위치를 파란색 구(SPHERE)로 RViz 시각화 (항상 퍼블리시)
"""
import math
from typing import List

import numpy as np
try:
    import cupy as cp  # GPU 가속 (optional)
except ModuleNotFoundError:  # pragma: no cover
    cp = np

from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Int32

# ────────────────────────── 공통 유틸 ──────────────────────────

def remap_angle(p):
    """0 ≤ θ < 2π 범위 각도 반환."""
    a = math.atan2(p[1], p[0])
    return a + 2 * math.pi if a < 0 else a


class MissionSubscriberMixin(Node):
    """`mission_state`(Int32)을 구독해서 값을 보관."""

    def __init__(self, name: str):
        super().__init__(name)
        self._mission_state = 0
        self.create_subscription(Int32, "mission_state", self._ms_cb, 10)

    def _ms_cb(self, msg: Int32):
        self._mission_state = msg.data

    # 퍼블리시 gating 용
    def in_mode_b(self) -> bool:
        return self._mission_state == 2


# ─────────────────────── 오른쪽 벽 노드 ────────────────────────
class RightWallBSplinePointsPublisher(MissionSubscriberMixin):
    def __init__(self):
        super().__init__("right_wall_bspline_points_publisher")
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.marker_pub = self.create_publisher(Marker, "right_wall_points", 10)
        self.get_logger().info("RightWallBSplinePointsPublisher started.")

    # --------------------------------------------------
    def scan_cb(self, scan_msg: LaserScan):
        self.process_scan(scan_msg)  # 항상 수행

    # --------------------------------------------------
    def process_scan(self, scan_msg: LaserScan):
        angle_min, angle_inc = scan_msg.angle_min, scan_msg.angle_increment
        ranges = scan_msg.ranges
        right_pts: List[List[float]] = []

        # 오른쪽 벽 후보: 90° ≤ θ ≤ 190° (센서 기준)
        for i, r in enumerate(ranges):
            if math.isinf(r) or math.isnan(r):
                continue
            ang = angle_min + i * angle_inc
            ang_deg = math.degrees(ang) % 360
            if 70 <= ang_deg <= 190:
                x, y = r * math.cos(ang), r * math.sin(ang)
                right_pts.append([x, y])
        if not right_pts:
            return
        pts_np = np.asarray(right_pts)

        # DBSCAN 클러스터링
        labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(pts_np)
        uniq = set(labels) - {-1}
        if not uniq:
            return

        cand = None
        for lab in uniq:
            cl = pts_np[labels == lab]
            if any(55 <= math.degrees(remap_angle(p)) <= 95 for p in cl):
                cand = cl if cand is None or cl.shape[0] > cand.shape[0] else cand
        if cand is None:
            lab = max(uniq, key=lambda l: (labels == l).sum())
            cand = pts_np[labels == lab]
        if cand.shape[0] < 3:
            return

        cand = cand[np.argsort([remap_angle(p) for p in cand])]
        x_vals, y_vals = cand[:, 0], cand[:, 1]
        try:
            tck, _ = splprep([x_vals, y_vals], s=0.01)
        except Exception as e:  # pragma: no cover
            self.get_logger().debug(f"Spline error: {e}")
            return

        u_dense = np.linspace(0, 1, 1000)
        dx, dy = splev(u_dense, tck)
        seg_len = np.hypot(np.diff(dx), np.diff(dy))
        cum = np.concatenate(([0.0], np.cumsum(seg_len)))
        if cum[-1] < 0.05:
            return
        u_samp = np.interp(np.linspace(0, cum[-1], int(cum[-1] / 0.05) + 1), cum, u_dense)
        sx, sy = splev(u_samp, tck)

        mk = Marker()
        mk.header.frame_id = scan_msg.header.frame_id or "base_link"
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = "right_wall_bspline"
        mk.type = Marker.POINTS
        mk.action = Marker.ADD
        mk.scale.x = mk.scale.y = 0.02
        mk.color.b = mk.color.a = 1.0
        mk.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in zip(sx, sy)]
        self.marker_pub.publish(mk)


# ──────────────────────── 왼쪽 벽 노드 ─────────────────────────
class LeftWallBSplinePointsPublisher(MissionSubscriberMixin):
    def __init__(self):
        super().__init__("left_wall_bspline_points_publisher")
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.marker_pub = self.create_publisher(Marker, "left_wall_points", 10)
        self.get_logger().info("LeftWallBSplinePointsPublisher started.")

    def scan_cb(self, scan_msg: LaserScan):
        self.process_scan(scan_msg)

    def process_scan(self, scan_msg: LaserScan):
        angle_min, angle_inc = scan_msg.angle_min, scan_msg.angle_increment
        pts: List[List[float]] = []
        for i, r in enumerate(scan_msg.ranges):
            if math.isinf(r) or math.isnan(r):
                continue
            ang = angle_min + i * angle_inc
            ang_deg = math.degrees(ang) % 360
            if 170 <= ang_deg <= 300:
                x, y = r * math.cos(ang), r * math.sin(ang)
                pts.append([x, y])
        if not pts:
            return
        pts_np = np.asarray(pts)
        labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(pts_np)
        uniq = set(labels) - {-1}
        if not uniq:
            return
        cand = None
        for lab in uniq:
            cl = pts_np[labels == lab]
            if any(265 <= math.degrees(remap_angle(p)) <= 275 for p in cl):
                cand = cl if cand is None or cl.shape[0] > cand.shape[0] else cand
        if cand is None:
            lab = max(uniq, key=lambda l: (labels == l).sum())
            cand = pts_np[labels == lab]
        if cand.shape[0] < 3:
            return
        cand = cand[np.argsort([remap_angle(p) for p in cand])[::-1]]
        x_vals, y_vals = cand[:, 0], cand[:, 1]
        try:
            tck, _ = splprep([x_vals, y_vals], s=0.01)
        except Exception:
            return
        u_dense = np.linspace(0, 1, 1000)
        dx, dy = splev(u_dense, tck)
        seg_len = np.hypot(np.diff(dx), np.diff(dy))
        cum = np.concatenate(([0.0], np.cumsum(seg_len)))
        if cum[-1] < 0.05:
            return
        u_samp = np.interp(np.linspace(0, cum[-1], int(cum[-1] / 0.05) + 1), cum, u_dense)
        sx, sy = splev(u_samp, tck)
        mk = Marker()
        mk.header.frame_id = scan_msg.header.frame_id or "base_link"
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = "left_wall_bspline"
        mk.type = Marker.POINTS
        mk.action = Marker.ADD
        mk.scale.x = mk.scale.y = 0.02
        mk.color.g = mk.color.a = 1.0
        mk.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in zip(sx, sy)]
        self.marker_pub.publish(mk)


# ──────────────────────── 경로 & 제어 노드 ─────────────────────
class FinalPathPublisher(MissionSubscriberMixin):
    def __init__(self):
        super().__init__("final_path_publisher")
        self.create_subscription(Marker, "left_wall_points", self.left_cb, 10)
        self.create_subscription(Marker, "right_wall_points", self.right_cb, 10)

        self.marker_pub    = self.create_publisher(Marker, "final_path_points", 10)
        self.lookahead_pub = self.create_publisher(Marker, "lookahead_point", 10)
        self.axle_pub      = self.create_publisher(Marker, "rear_axle_marker", 10)  # 후륜축 마커 퍼블리셔
        self.steering_pub  = self.create_publisher(Float32, "steering_angle", 10)
        self.speed_pub     = self.create_publisher(Int32,   "motor_speed",    10)

        self.left_marker  = None
        self.right_marker = None
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Pure Pursuit 파라미터
        self.L_vehicle = 0.2
        self.base_Ld   = 0.6
        self.max_Ld    = 1.2
        self.min_Ld    = 0.5
        self.rear_axle = np.array([0.2, 0.0])  # 차량 후륜축 위치 (base_link 기준)

        # PWM 범위
        self.max_pwm = 80
        self.min_pwm = 40

        # ★ 곡률-기반 속도 조절용 파라미터
        self.kappa_slow = 0.65   # [1/m]  (R ≈ 1 m 이하에서 최저 속도)

        self.get_logger().info("FinalPathPublisher started.")

    def left_cb(self, m: Marker):
        self.left_marker = m

    def right_cb(self, m: Marker):
        self.right_marker = m

    def timer_cb(self):
        # 벽 마커 준비 확인
        if self.left_marker is None or self.right_marker is None:
            return

        left_pts  = np.asarray([[p.x, p.y] for p in self.left_marker.points])
        right_pts = np.asarray([[p.x, p.y] for p in self.right_marker.points])
        if left_pts.shape[0] < 2 or right_pts.shape[0] < 2:
            return

        n_left, n_right = left_pts.shape[0], right_pts.shape[0]
        if n_left != n_right:
            idx = np.linspace(0, n_right - 1, n_left)
            right_pts = np.column_stack([
                np.interp(idx, np.arange(n_right), right_pts[:, i]) for i in range(2)
            ])

        mid1 = (left_pts[:-1] + right_pts[:-1]) / 2.0
        mid2 = (left_pts[:-1] + right_pts[1:]) / 2.0
        final_path = cp.asnumpy(cp.concatenate((cp.asarray(mid1), cp.asarray(mid2)), axis=0))

        # Marker: 최종 경로
        mk = Marker()
        mk.header.frame_id = self.left_marker.header.frame_id or "base_link"
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = "final_path"
        mk.type = Marker.POINTS
        mk.action = Marker.ADD
        mk.scale.x = mk.scale.y = 0.01
        mk.color.r = mk.color.g = mk.color.a = 1.0
        mk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in final_path]
        self.marker_pub.publish(mk)

        # Marker: Rear-Axle 위치 (항상 퍼블리시)
        ax_mk = Marker()
        ax_mk.header = mk.header
        ax_mk.ns = "rear_axle"
        ax_mk.type = Marker.SPHERE
        ax_mk.action = Marker.ADD
        ax_mk.scale.x = ax_mk.scale.y = ax_mk.scale.z = 0.05
        ax_mk.color.b = ax_mk.color.a = 1.0
        ax_mk.pose.position.x = float(self.rear_axle[0])
        ax_mk.pose.position.y = float(self.rear_axle[1])
        self.axle_pub.publish(ax_mk)

        # Pure Pursuit 계산
        Ld = max(self.min_Ld, min(self.max_Ld, self.base_Ld))
        dists = np.linalg.norm(final_path - self.rear_axle[:2], axis=1)
        idxs = np.where(dists >= Ld)[0]
        if idxs.size == 0:
            tgt = final_path[-1]
        else:
            tgt = final_path[idxs[0]]

        # Marker: look-ahead 포인트 (항상 퍼블리시)
        la_mk = Marker()
        la_mk.header = mk.header
        la_mk.ns = "lookahead"
        la_mk.type = Marker.SPHERE
        la_mk.action = Marker.ADD
        la_mk.scale.x = la_mk.scale.y = la_mk.scale.z = 0.05
        la_mk.color.r = la_mk.color.a = 1.0
        la_mk.pose.position.x = float(tgt[0])
        la_mk.pose.position.y = float(tgt[1])
        self.lookahead_pub.publish(la_mk)

        dx, dy = tgt - self.rear_axle[:2]
        alpha = math.atan2(dy, dx)
        delta = math.atan2(2.0 * self.L_vehicle * math.sin(alpha), Ld)
        wheel_deg = math.degrees(delta)

        # 곡률 κ (1/m)
        kappa = abs(2.0 * math.sin(alpha) / Ld)

        # κ → PWM 선형 매핑
        ratio = min(kappa / self.kappa_slow, 1.0)          # 0 ~ 1
        pwm   = int(self.max_pwm - ratio * (self.max_pwm - self.min_pwm))
        pwm   = max(self.min_pwm, min(self.max_pwm, pwm))  # saturate

        # ▸ angle & speed 퍼블리시는 mission_state == 2 일 때만
        if self.in_mode_b():
            self.steering_pub.publish(Float32(data=float(wheel_deg)))
            self.speed_pub.publish(Int32(data=pwm))


# ───────────────────────────── main ────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    right_node = RightWallBSplinePointsPublisher()
    left_node  = LeftWallBSplinePointsPublisher()
    path_node  = FinalPathPublisher()

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(right_node)
    executor.add_node(left_node)
    executor.add_node(path_node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        right_node.destroy_node()
        left_node.destroy_node()
        path_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

