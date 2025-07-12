# launch/mission1.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # ───────── 카메라 ─────────
        Node(
            package='mission1',
            executable='camera_node',
            name='camera',
            output='screen',
            parameters=[  # 필요하면 해상도·ID 등을 여기서 넘길 수 있습니다
                # {'device_id': 0},          # 예) 파라미터 사용 시
                # {'frame_rate': 30},
            ],
        ),

        # ───────── 차선 검출 ───────
        Node(
            package='mission1',
            executable='lane_detection_node',
            name='lane_detection',
            output='screen',
        ),

        # ───────── Pure-Pursuit ───
        Node(
            package='mission1',
            executable='pure_pursuit_node',
            name='pure_pursuit',
            output='screen',
        ),
    ])

