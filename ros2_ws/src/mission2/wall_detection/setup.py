from glob import glob
from setuptools import find_packages, setup

package_name = 'wall_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament 패키지 인덱스 등록용
        ('share/ament_index/resource_index/packages', [
            'resource/' + package_name
        ]),

        # 패키지 메타정보
        ('share/' + package_name, [
            'package.xml'
        ]),

        # launch 디렉터리 안의 .launch.py 파일 설치
        ('share/' + package_name + '/launch', 
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoo',
    maintainer_email='yoo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wall_detection = wall_detection.wall_detection:main',
            'wall_detection2 = wall_detection.wall_detection2:main',
            'wall_detection3 = wall_detection.wall_detection3:main',
        ],
    },
)

