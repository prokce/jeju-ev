import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt

clicked_points = []
fig, ax = plt.subplots()

# === 좌표계 변환 관련 상수 ===
PIXEL_TO_METER = 0.0028     # scale: 픽셀당 거리 (m/pixel)
CAMERA_TO_REAR_AXLE = 0.2    # 카메라 → 차량 후륜축 거리 (m)
BEV_BOTTOM_FROM_CAMERA = 0.25  # 카메라 기준 BEV 하단 거리 (m)
OFFSET_X = CAMERA_TO_REAR_AXLE + BEV_BOTTOM_FROM_CAMERA  # = 0.45m

# === 경로 생성 관련 상수 ===
MIN_LANE_POINTS = 4             # 경로 생성을 위한 최소 차선 점 개수
LANE_OFFSET_DEFAULT = 0.8       # 한쪽 차선만 검출 시 적용할 lateral offset 거리 (m)




### === hsv기반 흰색 마스킹 + 모폴로지 연산 + 가우시안 필터링 ===
def filter_white_lane(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white, upper_white = (0, 0, 180), (180, 6, 255)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 구멍 메우기
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 잡음 제거
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)

### === 버드아이뷰 변환 ===
def get_birds_eye_view(image):
    height, width = image.shape[:2]
    src = np.float32([
    [width * 0.24, height * 0.1],   # top-left
    [width * 0.76, height * 0.1],   # top-right
    [width * 1.0,  height * 0.8],   # bottom-right
    [width * 0.0,  height * 0.8]    # bottom-left
    ])
    dst = np.float32([
    [width * 0.1, 0],               # top-left
    [width * 0.9, 0],               # top-right
    [width * 0.9, height],          # bottom-right
    [width * 0.1, height]           # bottom-left
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    bev = cv2.warpPerspective(image, M, (width, height))
    return bev, src

### === 히스토그램 기반 초기 차선 추출 + 화면 중앙 & 하단 1/4 기준 좌우차선 분리 ===
def find_lane_peaks(mask, width, height):
    bottom_region = mask[int(height * 1 / 2):, :]  # 하단 33%만 선택
    hist = np.sum(bottom_region, axis=0)         
    threshold = 0.1 * np.max(hist)               # 최대값의 25% 이상만 유의미한 피크로 간주
    peaks = np.where(hist > threshold)[0]         # threshold 이상인 x좌표들 추출

    # 인접한 x좌표들끼리 클러스터링 (100픽셀(0.333m) 이하 간격이면 같은 클러스터)
    clusters, temp = [], []
    for i, p in enumerate(peaks):
        if i == 0 or p - peaks[i - 1] <= 50:
            temp.append(p)
        else:
            clusters.append(temp)                
            temp = [p]
    if temp:
        clusters.append(temp)                     

    valid_clusters = []
    for cluster in clusters:
        mean_x = int(np.mean(cluster))
        # 중심 x 기준으로 좌우 ±10픽셀 범위에서 하단 1/4 높이만큼 잘라낸 영역 추출
        region = mask[int(height * 1 / 2):, mean_x - 10: mean_x + 10]
        # 해당 영역에 255(흰색) 픽셀이 5개 이상이면 유효한 차선 클러스터로 판단
        if np.sum(region) > 5:
            valid_clusters.append(cluster)

    midpoint = width // 2
    lanes = {'left': None, 'right': None}
    for cluster in valid_clusters:
        mean_x = int(np.mean(cluster))
        if mean_x < midpoint and lanes['left'] is None:
            lanes['left'] = mean_x               
        elif mean_x >= midpoint and lanes['right'] is None:
            lanes['right'] = mean_x      

    return lanes

### === 슬라이딩 윈도우 기반 다음 차선 추출 + 연결 거리 제한 (plot) 기존 link_thr=40===
def sliding_window_points(bin_img, base_x, nwindows=30, margin=100, minpix=20, link_thr=40):
    h, w = bin_img.shape
    nonzero_y, nonzero_x = bin_img.nonzero()
    win_h = h // nwindows
    points = []
    x_cur = base_x
    for i in range(nwindows):
        y_low, y_high = h - (i+1)*win_h, h - i*win_h
        x_low, x_high = x_cur - margin, x_cur + margin
        good = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                (nonzero_x >= x_low) & (nonzero_x < x_high)).nonzero()[0]
        if len(good) > minpix:
            x_mean = int(np.mean(nonzero_x[good]))
            y_mean = int(np.mean(nonzero_y[good]))
            # 연결 거리 제한 (커브 급변 방지)
            if not points or np.hypot(x_mean - points[-1][0],
                                      y_mean - points[-1][1]) <= link_thr:
                points.append((x_mean, y_mean))
                x_cur = x_mean
            else:
                # 거리 초과 → 해당 윈도우 무시 (추적 중단 X)
                continue
    return points

### === 차량좌표계 변환(차량 후륜축 기준) ===
def convert_to_vehicle_coords(x_pixel, y_pixel, image_width, image_height):
    dx = OFFSET_X + (image_height - y_pixel) * PIXEL_TO_METER   # 전방 +x
    dy = -(x_pixel - image_width / 2) * PIXEL_TO_METER          # 좌측 +y
    return dx, dy

### === 3차 다항 피팅 ===
def fit_polynomial_path(path, order=3):
    if not path or len(path) < order + 1:
        return None
    xs, ys = zip(*path)
    try:
        coeffs = np.polyfit(xs, ys, order)
        return np.poly1d(coeffs)
    except Exception as e:
        print(f"[Polyfit] Error fitting polynomial: {e}")
        return None
    
### === 한쪽 차선만 검출 시 경로 생성 (예외처리) ===
import numpy as np

import numpy as np

def generate_offset_path_from_single_lane(
    lane_points, image_width, image_height,
    is_left=True, lateral_offset=LANE_OFFSET_DEFAULT
):
    """
    한쪽 차선만 검출된 경우:
    각 점마다 국소적인 법선 방향을 계산하여 오프셋 경로를 생성합니다.
    i==0, i==n-1 에서도 중앙차분(central difference)을 사용해 끝점 스파이크를 감소시킵니다.
    - lane_points: 이미지 좌표계상의 (x, y) 리스트
    - image_width, image_height: 이미지 크기
    - is_left: 좌측 차선(True)인지 우측(False)인지
    - lateral_offset: 차량 중심선으로부터의 횡방향 오프셋 거리 (m)
    """
    n = len(lane_points)
    if n < 2:
        return []

    # 1) 모든 점을 차량 좌표계로 변환
    veh_points = [
        convert_to_vehicle_coords(px, py, image_width, image_height)
        for px, py in lane_points
    ]

    offset_path = []
    offset_sign = -1.0 if is_left else 1.0

    # 2) 각 점마다 개선된 차분 방식으로 접선/법선 계산
    for i, (vx, vy) in enumerate(veh_points):
        # dx, dy 초기화
        if n >= 3:
            if i == 0:
                # 두 칸 앞 (p2 - p0)
                dx = veh_points[2][0] - vx
                dy = veh_points[2][1] - vy
            elif i == n - 1:
                # 두 칸 뒤 (p_{n-1} - p_{n-3})
                dx = vx - veh_points[n - 3][0]
                dy = vy - veh_points[n - 3][1]
            else:
                # 중앙차분 (p_{i+1} - p_{i-1})
                dx = veh_points[i + 1][0] - veh_points[i - 1][0]
                dy = veh_points[i + 1][1] - veh_points[i - 1][1]
        else:
            # 점이 2개일 때는 기존 forward/backward diff
            prev_idx = max(i - 1, 0)
            next_idx = min(i + 1, n - 1)
            dx = veh_points[next_idx][0] - veh_points[prev_idx][0]
            dy = veh_points[next_idx][1] - veh_points[prev_idx][1]

        # 법선(–dy, dx) 계산 및 정규화
        length = np.hypot(dx, dy)
        if length == 0:
            perp_x, perp_y = 0.0, 0.0
        else:
            perp_x = -dy / length
            perp_y = dx / length

        # 횡방향 오프셋 적용
        cx = vx + offset_sign * lateral_offset * perp_x
        cy = vy + offset_sign * lateral_offset * perp_y
        offset_path.append((cx, cy))

    # 3) 전방(x) 기준 오름차순 정렬
    offset_path.sort(key=lambda p: p[0])
    return offset_path


### === 좌우 차선 검출 시 경로 생성 (예외처리 포함) === 
def path_planning_from_lanes(left_points, right_points, image_width, image_height):
    path = []

    def to_vehicle(p):
        return convert_to_vehicle_coords(p[0], p[1], image_width, image_height)

    len_left = len(left_points) if left_points else 0
    len_right = len(right_points) if right_points else 0

    # --- Case 1: 양쪽 차선 모두 있고, 둘 다 충분한 점이 있는 경우 ---
    if left_points and right_points and len_left >= MIN_LANE_POINTS and len_right >= MIN_LANE_POINTS:
        min_len = min(len_left, len_right)
        for i in range(min_len):
            lx, ly = to_vehicle(left_points[i])
            rx, ry = to_vehicle(right_points[i])
            path.append(((lx + rx) / 2, (ly + ry) / 2))

    # --- Case 2: 왼쪽만 충분한 경우 → 왼쪽 기반 offset ---
    elif left_points and len_left >= MIN_LANE_POINTS:
        path = generate_offset_path_from_single_lane(
            left_points, image_width, image_height, is_left=True, lateral_offset=LANE_OFFSET_DEFAULT)

    # --- Case 3: 오른쪽만 충분한 경우 → 오른쪽 기반 offset ---
    elif right_points and len_right >= MIN_LANE_POINTS:
        path = generate_offset_path_from_single_lane(
            right_points, image_width, image_height, is_left=False, lateral_offset=LANE_OFFSET_DEFAULT)

    else:
        path = []

    return path



### === imshow 시각화 함수 ===
def visualize_lanes_with_sliding_windows(mask, lanes, left_points, right_points, nwindows=30, margin=100):
    h, w = mask.shape
    win_h = h // nwindows
    vis_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    def draw_windows(points, base_x, color, point_color):
        x_cur = base_x
        for i in range(nwindows):
            y_low = h - (i + 1) * win_h
            y_high = h - i * win_h
            x_low = x_cur - margin
            x_high = x_cur + margin

            # 현재 윈도우 안에 들어온 점들
            pts_in_win = [pt for pt in points if y_low <= pt[1] < y_high and x_low <= pt[0] < x_high]

            if pts_in_win:
                # 윈도우 시각화 (조건부)
                cv2.rectangle(vis_img,
                              (max(0, x_low), y_low),
                              (min(w, x_high), y_high),
                              color, 2)
                # 다음 윈도우 기준 위치 갱신
                x_cur = int(np.mean([p[0] for p in pts_in_win]))
            else:
                break  # 더 이상 점이 없으면 윈도우도 중단

        # 검출된 점 시각화
        for (x, y) in points:
            cv2.circle(vis_img, (x, y), 4, point_color, -1)

    if lanes.get('left') is not None and left_points:
        draw_windows(left_points, lanes['left'], (255, 0, 0), (255, 0, 0))  # 파란색 (BGR)
    if lanes.get('right') is not None and right_points:
        draw_windows(right_points, lanes['right'], (0, 0, 255), (0, 0, 255))  # 빨간색 (BGR)

    return vis_img



### === matplotlib 시각화 함수 (후륜축 기준)===
def plot_path_matplotlib(path_points, left_points=None, right_points=None, image_width=None, image_height=None):
    if not path_points:
        return
    x_vals = [pt[0] for pt in path_points]
    y_vals = [pt[1] for pt in path_points]
    ax.clear()
    ax.plot(x_vals, y_vals, 'y-', linewidth=2, label='Fitted Path')
    if left_points:
        lx, ly = [], []
        for px, py in left_points:
            dx, dy = convert_to_vehicle_coords(px, py, image_width, image_height)
            lx.append(dx)
            ly.append(dy)
        ax.plot(lx, ly, 'bo-', markersize=4, label='Left Lane')
    if right_points:
        rx, ry = [], []
        for px, py in right_points:
            dx, dy = convert_to_vehicle_coords(px, py, image_width, image_height)
            rx.append(dx)
            ry.append(dy)
        ax.plot(rx, ry, 'ro-', markersize=4, label='Right Lane')
    ax.set_xlabel('X (forward from rear axle, m)')
    ax.set_ylabel('Y (left/right, m)')
    ax.set_title('Vehicle Coordinate View')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    plt.pause(0.001)

### === 퍼블리시 함수 ===
def publish_path(node, path_points, frame_id="vehicle"):
    path_msg = Path()
    path_msg.header.stamp = node.get_clock().now().to_msg()
    path_msg.header.frame_id = frame_id
    for x, y in path_points:
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        path_msg.poses.append(pose)
    node.path_pub.publish(path_msg)

### === 마우스 클릭으로 픽셀 간 거리 측정 ===
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"[CLICK] ({x}, {y})")
        if len(clicked_points) == 2:
            dx = clicked_points[1][0] - clicked_points[0][0]
            dy = clicked_points[1][1] - clicked_points[0][1]
            dist = np.hypot(dx, dy)
            print(f"📏 Pixel distance: {dist:.2f} px")
            clicked_points.clear()

### === 마우스 클릭으로 hsv 측정 ===
def hsv_mouse_callback(event, x, y, flags, param):
    """
    마우스 클릭 시 해당 픽셀의 HSV 값 출력
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_bgr = param['frame']
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        pixel_hsv = hsv[y, x]
        print(f"[HSV] x={x}, y={y} → H: {pixel_hsv[0]}, S: {pixel_hsv[1]}, V: {pixel_hsv[2]}")


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.bridge = CvBridge()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos
        )
        self.path_pub = self.create_publisher(Path, '/lane/path', 10)

    def image_callback(self, msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        lane_mask = filter_white_lane(frame)
        bev, src_points = get_birds_eye_view(lane_mask)
        bev_height, bev_width = bev.shape[:2]
        lanes = find_lane_peaks(bev, bev_width, bev_height)

        left_points = sliding_window_points(bev, lanes['left']) if lanes['left'] is not None else []
        right_points = sliding_window_points(bev, lanes['right']) if lanes['right'] is not None else []

        raw_path = path_planning_from_lanes(left_points, right_points, bev_width, bev_height)
        poly = fit_polynomial_path(raw_path)
        if poly:
            min_x = min(x for x, _ in raw_path)
            max_x = max(x for x, _ in raw_path)
            step = 0.1
            fitted_path = [(x, poly(x)) for x in np.arange(min_x, max_x, step)]
        else:
            fitted_path = raw_path

        if left_points and right_points:
            lane_info = "both"
        elif left_points:
            lane_info = "left_only"
        elif right_points:
            lane_info = "right_only"
        else:
            lane_info = "none"
        #self.get_logger().info(f"📋 Detected lanes: {lane_info.upper()} (L: {len(left_points)}, R: {len(right_points)})")

        publish_path(self, fitted_path, frame_id=lane_info)

        frame_vis = frame.copy()
        pts = src_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for (x, y) in src_points:
            cv2.circle(frame_vis, (int(x), int(y)), 5, (0, 0, 255), -1)

        plot_path_matplotlib(
            fitted_path,
            left_points=left_points,
            right_points=right_points,
            image_width=bev_width,
            image_height=bev_height
        )

        vis_bev = visualize_lanes_with_sliding_windows(
            mask=bev,
            lanes=lanes,
            left_points=left_points,
            right_points=right_points,
            nwindows=30,
            margin=100
        )

        resized_bev = cv2.resize(vis_bev, (640, 480))
        cv2.imshow("Bird's Eye View", resized_bev)
        cv2.setMouseCallback("Bird's Eye View", mouse_callback)
        resized_frame = cv2.resize(frame_vis, (640, 480))
        cv2.imshow("Input Frame with BEV ROI", resized_frame)
        cv2.setMouseCallback("Input Frame with BEV ROI", hsv_mouse_callback, param={'frame': frame})  # HSV 측정용
        cv2.waitKey(1)



def main(args=None):
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()
    rclpy.init(args=args)
    rclpy.spin(LaneDetectionNode())
    rclpy.shutdown()

