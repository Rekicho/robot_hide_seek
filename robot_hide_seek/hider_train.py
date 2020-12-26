from math import radians, degrees, isinf, pi, inf

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from robot_hide_seek.utils import *

class HiderTrain(Node):
    follow_id = inf
    follow_distance = inf
    follow_angle = inf
    time = -1
    lidar_sensors = []
    result = 0
    
    def __init__(self, id):
        super().__init__('hider')

        self.node_topic = '/hider_' + str(id)

        self.game_sub = self.create_subscription(
            String,
            self.node_topic + '/game',
            self.game_callback,
            10
        )

        self.clock_sub = self.create_subscription(
            Clock, 
            '/clock', 
            self.clock_callback,
            10
        )
        self.vel_pub = self.create_publisher(
            Twist,
            self.node_topic + '/cmd_vel',
            10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, 
            self.node_topic + '/scan', 
            self.lidar_callback,
            qos_profile_sensor_data
        )

    def reset(self):
        self.follow_id = inf
        self.follow_distance = inf
        self.follow_angle = inf
        self.lidar_sensors = []

    def clock_callback(self, msg):
        if int(msg.clock.sec) < self.time:
            self.result = 0
            self.reset()

        self.time = int(msg.clock.sec)

    def game_callback(self, msg):
        if msg.data == START_MSG:
            return

        elif msg.data == GAMEOVER_MSG:
            self.endgame()

        message = msg.data.rstrip().split('\n\n')

        if message[0] == POSITIONS_MSG_HEADER:
            angles = [float(pos.split('\n')[0][7:]) for pos in message[1:]]
            distances = [float(pos.split('\n')[1][10:]) for pos in message[1:]]

            closest = (inf, inf)

            for i, dist in enumerate(distances):
                if dist < closest[1]:
                    closest = (i, dist)

            if not isinf(closest[0]):
                self.follow_id = closest[0]
                self.follow_angle = angles[self.follow_id]
                self.follow_distance = closest[1]

    def lidar_callback(self, msg):
        self.lidar_sensors = msg.ranges[:]

    def endgame(self):
        if self.time < GAME_TIME_LIMIT:
            self.result = -1

        else:
            self.result = 1

        self.reset()