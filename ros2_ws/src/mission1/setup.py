from setuptools import find_packages, setup
from glob import glob          # ← 추가

package_name = 'mission1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch/*.launch.py를 설치에 포함
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoo',
    maintainer_email='yoo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'lane_detection_node = mission1.lane_detection_node:main',
            'pure_pursuit_node  = mission1.pure_pursuit_node:main',
            'camera_node        = mission1.camera_node:main',   # ← 추가
        ],
    },
)

