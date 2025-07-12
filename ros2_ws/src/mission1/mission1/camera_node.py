#!/usr/bin/env python3
# camera_node_optimized.py ── 최대 화각 + 리사이즈 + 저지연 퍼블리셔

import cv2
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage


class CameraNodeOptimized(Node):
    """
    • V4L2 + MJPG 1920×1080 @30 FPS
    • 센서 전체 화각 확보
    • 1280×720 리사이즈 후 압축 → 속도 향상
    • JPEG 60% 압축 → 인코딩 부하 최소화
    • BEST_EFFORT QoS depth 1 → 지연 없음
    """

    def __init__(self):
        super().__init__('camera_node_optimized')

        # ───── QoS 설정: 최신 프레임 1장, 손실 허용
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.pub = self.create_publisher(
            CompressedImage, '/camera/image_raw/compressed', qos
        )

        # ───── 카메라 열기
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("❌ /dev/video4 열기 실패")
            raise RuntimeError("VideoCapture open failed")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # ───── 캡처 스레드 시작
        threading.Thread(target=self._capture_and_publish_loop, daemon=True).start()

    # ──────────────────────────────────────────────────────────────
    # 캡처 + 리사이즈 + 압축 → 퍼블리시 루프
    # ──────────────────────────────────────────────────────────────
    def _capture_and_publish_loop(self):
        while rclpy.ok():
            for _ in range(1):  # 내부 버퍼 비우기
                self.cap.grab()

            ret, frame = self.cap.read()
            if not ret:
                continue

            # (1) 해상도 축소 (화각 유지 + 성능 확보)
            frame = cv2.resize(frame, (1280, 720))

            # (2) JPEG 압축률 60%
            ok, enc = cv2.imencode('.jpg', frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not ok:
                self.get_logger().warn("⚠️ JPEG 인코딩 실패")
                continue

            # (3) ROS 메시지 생성 및 퍼블리시
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = 'jpeg'
            msg.data = enc.tobytes()
            self.pub.publish(msg)

    # ───── 종료 처리
    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNodeOptimized()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

