from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mission_state',           
            executable='mission_state',      
            name='mission_state',
            output='screen',
        ),
    ])

