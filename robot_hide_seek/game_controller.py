import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from rosgraph_msgs.msg import Clock

from robot_hide_seek.utils import START_MSG

SECONDS_HIDER_START = 0
SECONDS_SEEKER_START = 5

class HideSeek(Node):
    
    def __init__(self):
        super().__init__('hide_seek')
        
        self.hider_started = False
        self.seeker_started = False

        self.clock_sub = self.create_subscription(
            Clock, 
            '/clock', 
            self.clock_callback,
            10)
        self.hider_pub = self.create_publisher(
            String,
            '/hider/game',
            10
        )
        self.seeker_pub = self.create_publisher(
            String,
            '/seeker/game',
            10
        )

    def clock_callback(self, msg):
        if msg.clock.sec == SECONDS_HIDER_START and not self.hider_started:
            self.start_hider()

        if msg.clock.sec == SECONDS_SEEKER_START and not self.seeker_started:
            self.start_seeker()

    def start_hider(self):
        msg = String()
        msg.data = START_MSG
        
        self.hider_pub.publish(msg)
        hider_started = True

    def start_seeker(self):
        msg = String()
        msg.data = START_MSG
        
        self.seeker_pub.publish(msg)
        seeker_started = True
        

def main(args=None):
    rclpy.init(args=args)

    hide_seek = HideSeek()

    rclpy.spin(hide_seek)

    hide_seek.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
