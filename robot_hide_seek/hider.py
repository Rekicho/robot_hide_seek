from math import radians, degrees, isinf

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class HideSeek(Node):
    
    def __init__(self):
        super().__init__('hide_seek')
        self.subscription = self.create_subscription(
            LaserScan, 
            '/hider/scan', 
            self.lidar_callback,
            qos_profile_sensor_data)
        self.publisher = self.create_publisher(
            Twist,
            '/hider/cmd_vel',
            10
        )

    def lidar_callback(self, msg):
        min_range = msg.ranges[0]
        min_angle = msg.angle_min

        for i in range(1, len(msg.ranges)):
            angle = degrees(msg.angle_min + (i* msg.angle_increment))

            if msg.ranges[i] < min_range:
                min_range = msg.ranges[i]
                min_angle = angle

        print("Angle: " + str(min_angle) + "Range: " + str(min_range))

        if isinf(min_range):
            return

        vel = Twist()
        vel.linear.x = 0.1
        
        if min_angle < 180:
            vel.angular.z = -radians(abs(min_angle - 360)) * 0.25

        else:
            vel.angular.z = radians(min_angle) * 0.25

        self.publisher.publish(vel)

def main(args=None):
    rclpy.init(args=args)

    hide_seek = HideSeek()

    rclpy.spin(hide_seek)

    hide_seek.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
