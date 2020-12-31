import os
from glob import glob
from setuptools import setup

package_name = 'robot_hide_seek'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.model')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rekicho',
    maintainer_email='rekicho@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hider = robot_hide_seek.hider:main',
            'seeker = robot_hide_seek.seeker:main',
            'game_controller = robot_hide_seek.game_controller:main',
            'train_hider = robot_hide_seek.train_hider:main',
            'train_seeker = robot_hide_seek.train_seeker:main',
            'deep_train = robot_hide_seek.deep_train:main' #not working
        ],
    },
)
