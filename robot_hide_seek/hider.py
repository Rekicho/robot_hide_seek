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
            'hider/cmd_vel',
            10
        )

    def lidar_callback(self, msg):
        vel = Twist()
        vel.linear.x = 1.0
        self.publisher.publish(vel)

def main(args=None):
    rclpy.init(args=args)

    hide_seek = HideSeek()

    rclpy.spin(hide_seek)

    hide_seek.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
