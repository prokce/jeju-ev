from setuptools import find_packages, setup

package_name = 'mission3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoo',
    maintainer_email='yoo@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_node = mission3.odom:main',
            'odom_node2 = mission3.odem2:main',
            'odom_node3 = mission3.odem3:main',
            'odom_node4 = mission3.odem4:main',
            'odom_node5 = mission3.odem5:main',
            'odom_node6 = mission3.odem6:main'
        ],
    },
)
