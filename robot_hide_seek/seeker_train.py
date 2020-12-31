from math import radians, degrees, isinf, pi, inf

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from robot_hide_seek.utils import *

class SeekerTrain(Node):
    follow_id = inf
    follow_distance = inf
    follow_angle = inf
    angles = []
    distances = []
    time = -1
    lidar_sensors = []
    result = 0

    def __init__(self, id):
        super().__init__('seeker_' + str(id))

        self.node_topic = '/seeker_' + str(id)

        self.game_sub = self.create_subscription(
            String,
            self.node_topic + '/game',
            self.game_callback,
            10
        )
        self.seeker_coord_sub = self.create_subscription(
            String,
            '/seekers',
            self.coord_callback,
            10
        )
        self.seeker_coord_pub = self.create_publisher(
            String,
            '/seekers',
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
        self.angles = []
        self.distances = []
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
            self.angles = [float(pos.split('\n')[0][7:]) for pos in message[1:]]
            self.distances = [float(pos.split('\n')[1][10:]) for pos in message[1:]]

            if self.seeker_coord_pub:
                self.share_distances()

    def share_distances(self):
        msg = String()

        for dist in self.distances:
            msg.data += str(dist) + '\n'

        self.seeker_coord_pub.publish(msg)

    def coord_callback(self, msg):
        if self.time < SECONDS_SEEKER_START:
            return
        other_distances = msg.data.rstrip().split('\n')

        if len(other_distances) > len(self.distances):
            return
        
        min_difference = (inf, inf)

        for i, distance in enumerate(other_distances):
            diff = self.distances[i] - float(distance)

            if diff < min_difference[1]:
                min_difference = (i, diff)

        if not isinf(min_difference[0]):
            self.follow_id = min_difference[0]
            self.follow_angle = self.angles[min_difference[0]]
            self.follow_distance = self.distances[min_difference[0]]

    def lidar_callback(self, msg):
        self.lidar_sensors = [msg.ranges[0], msg.ranges[45], msg.ranges[90], msg.ranges[135], msg.ranges[180], msg.ranges[225], msg.ranges[270], msg.ranges[315]]

    def endgame(self):
        if self.time < SECONDS_SEEKER_START:
            self.result = 0

        elif self.time < GAME_TIME_LIMIT:
            self.result = 1

        else:
            self.result = -1

        self.reset()