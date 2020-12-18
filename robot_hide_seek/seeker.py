from math import radians, degrees, isinf, pi

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from robot_hide_seek.utils import *

class Seeker(Node):
    follow_angle = 0
    
    def __init__(self):
        super().__init__('seeker')

        self.declare_parameter('id')
        id = self.get_parameter('id').value
        self.node_topic = '/seeker_' + str(id)

        self.game_sub = self.create_subscription(
            String,
            self.node_topic + '/game',
            self.game_callback,
            10
        )

    def game_callback(self, msg):
        if msg.data == START_MSG:
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
            return

        elif msg.data == GAMEOVER_MSG:
            self.endgame()

        message = msg.data.split(' ')

        if message[0] == 'Angle':
            self.follow_angle = float(message[1])

    def lidar_callback(self, msg):
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
        vel.linear.x = SEEKER_LINEAR_SPEED

        #Temporary
        if min_range <= MIN_DISTANCE_TO_WALL:
            if min_angle < 5 * pi / 8:
                if min_angle < pi / 8:
                    vel.linear.x = -SPEED_NEAR_WALL
                elif min_angle < 3 * pi / 8:
                    vel.linear.x = SPEED_NEAR_WALL

                if self.follow_angle >= 0:
                    min_angle -= 5 * pi / 8
                else:
                    min_angle += 5 * pi / 8

            elif min_angle > 11 * pi / 8:
                if min_angle > 15 * pi / 8:
                    vel.linear.x = -SPEED_NEAR_WALL
                elif min_angle > 13 * pi / 8:
                    vel.linear.x = SPEED_NEAR_WALL
                min_angle -= 11 * pi / 8

                if self.follow_angle >= 0:
                    min_angle -= 11 * pi / 8
                else:
                    min_angle += 11 * pi / 8
        else:
            min_angle = self.follow_angle / TURN_RATIO

        vel.angular.z = min_angle * TURN_RATIO

        self.vel_pub.publish(vel)

    def endgame(self):
        self.vel_pub.publish(Twist())
        exit()

def main(args=None):
    rclpy.init(args=args)

    seeker = Seeker()

    rclpy.spin(seeker)

    seeker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
