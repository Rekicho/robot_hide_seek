from math import radians, degrees, isinf

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from robot_hide_seek.utils import START_MSG

class Seeker(Node):
    
    def __init__(self):
        super().__init__('seeker')
        self.game_sub = self.create_subscription(
            String,
            '/seeker/game',
            self.game_callback,
            10
        )

    def game_callback(self, msg):
        if msg.data == START_MSG:
            self.vel_pub = self.create_publisher(
                Twist,
                '/seeker/cmd_vel',
                10
            )
            self.lidar_sub = self.create_subscription(
                LaserScan, 
                '/seeker/scan', 
                self.lidar_callback,
                qos_profile_sensor_data
            )

    def lidar_callback(self, msg):
        min_range = msg.ranges[0]
        min_angle = msg.angle_min

        for i in range(1, len(msg.ranges)):
            angle = degrees(msg.angle_min + (i* msg.angle_increment))

            if msg.ranges[i] < min_range:
                min_range = msg.ranges[i]
                min_angle = angle

        if isinf(min_range):
            return

        vel = Twist()
        vel.linear.x = 0.2
        
        if min_angle < 180:
            vel.angular.z = radians(abs(min_angle - 360)) * 0.25

        else:
            vel.angular.z = -radians(min_angle) * 0.25

        self.vel_pub.publish(vel)

def main(args=None):
    rclpy.init(args=args)

    seeker = Seeker()

    rclpy.spin(seeker)

    seeker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
