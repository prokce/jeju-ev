from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wall_detection',           
            executable='wall_detection2',      
            name='wall_detection',
            output='screen',
        ),
    ])




