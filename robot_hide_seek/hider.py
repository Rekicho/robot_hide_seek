from math import radians, degrees, isinf, pi, inf

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from robot_hide_seek.utils import *

class Hider(Node):
    follow_id = inf
    follow_distance = inf
    follow_angle = inf
    time = -1
    gameover = True
    
    def __init__(self):
        super().__init__('hider')

        self.declare_parameter('id')
        id = self.get_parameter('id').value
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

    def clock_callback(self, msg):
        if int(msg.clock.sec) < self.time:
            self.gameover = False
            self.reset()

        self.time = int(msg.clock.sec)

    def game_callback(self, msg):
        if msg.data == START_MSG:
            return

        if msg.data == GAMEOVER_MSG:
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
        if self.time < SECONDS_HIDER_START or self.gameover:
            return

        min_range = msg.ranges[0]
        min_angle = msg.angle_min

        for i in range(1, len(msg.ranges)):
            angle = degrees(msg.angle_min + (i* msg.angle_increment))

            if msg.ranges[i] < min_range:
                min_range = msg.ranges[i]
                min_angle = radians(angle)

        if isinf(min_range):
            return

        vel = Twist()
        vel.linear.x = HIDER_LINEAR_SPEED

        #Temporary
        if min_range <= MIN_DISTANCE_TO_WALL:
            if min_angle < 5 * pi / 8:
                if min_angle < pi / 8:
                    vel.linear.x = -SPEED_NEAR_WALL
                elif min_angle < 3 * pi / 8:
                    vel.linear.x = SPEED_NEAR_WALL
                
                if not isinf(self.follow_angle):
                    if self.follow_angle >= 0:
                        min_angle += 5 * pi / 8
                    else:
                        min_angle -= 5 * pi / 8
                    
            elif min_angle > 11 * pi / 8:
                if min_angle > 15 * pi / 8:
                    vel.linear.x = -SPEED_NEAR_WALL
                elif min_angle > 13 * pi / 8:
                    vel.linear.x = SPEED_NEAR_WALL
                
                if not isinf(self.follow_angle):
                    if self.follow_angle >= 0:
                        min_angle += 11 * pi / 8
                    else:
                        min_angle -= 11 * pi / 8
        else:
            if not isinf(self.follow_angle):
                min_angle = self.follow_angle

            if min_angle > 0:
                min_angle -= pi
            else:
                min_angle += pi
            min_angle /= TURN_RATIO
        
        vel.angular.z = min_angle * TURN_RATIO

        self.vel_pub.publish(vel)

    def endgame(self):  
        self.gameover = True      
        self.vel_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)

    hider = Hider()

    rclpy.spin(hider)

    hider.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
