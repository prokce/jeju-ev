lidar="cd ~/ros2_ws && source ~/ros2_ws/install/setup.bash && sudo chmod 777 /dev/ttyUSB0 && ros2 launch sllidar_ros2 view_sllidar_a3_launch.py"
# mission1 launch alias
cam="v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=2"
1="cd ~/ros2_ws && source install/setup.bash && ros2 launch mission1 mission1.launch.py"
# mission2 launch alias
2="cd ~/ros2_ws && source install/setup.bash && ros2 launch wall_detection mission2.launch.py"
# mission3 launch alias
3="cd ~/ros2_ws && source install/setup.bash && ros2 run mission3 odom_node6"
# mission_state launch alias
mission="cd ~/ros2_ws && source install/setup.bash && ros2 launch mission_state mission_state.launch.py"
# micro-ROS 에이전트 실행용 alias
microros="cd ~/ws_test_microros && source ~/ws_test_microros/install/local_setup.bash && ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200"


#Board
Arduino Giga R1

#Arduino library
I2Cdev
micro_ros_arduino
MPU9250
MPU9250_DMP6
SparkFun_MPU-9250_Digital_Motion_Processing__DMP__Arduino_Librar
