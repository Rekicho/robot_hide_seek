import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry

from math import atan2, pi, sqrt

from robot_hide_seek.utils import *

class HideSeek(Node):
    
    def __init__(self):
        super().__init__('hide_seek')
        
        self.hider_started = False
        self.seeker_started = False
        self.hider_pos = [0, 0, 0]
        self.seeker_pos = [0, 0, 0]
        self.hider_yaw = 0
        self.seeker_yaw = 0

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
        self.hider_pos_sub = self.create_subscription(
            Odometry,
            '/hider/odom',
            self.hider_pos_callback,
            10
        )
        self.seeker_pos_sub = self.create_subscription(
            Odometry,
            '/seeker/odom',
            self.seeker_pos_callback,
            10
        )

    def clock_callback(self, msg):
        if msg.clock.sec == SECONDS_HIDER_START and not self.hider_started:
            self.start_hider()

        if msg.clock.sec == SECONDS_SEEKER_START and not self.seeker_started:
            self.start_seeker()

        if msg.clock.sec == GAME_TIME_LIMIT:
            self.endgame('Hider wins')

    def publish_str_msg(self, publisher, msg_data):
        msg = String()
        msg.data = msg_data
        publisher.publish(msg)

    def start_hider(self):
        print("START HIDER")
        self.hider_started = True
        self.publish_str_msg(self.hider_pub,START_MSG)

    def start_seeker(self):
        print("START SEEKER")
        self.seeker_started = True
        self.publish_str_msg(self.seeker_pub,START_MSG)

    def hider_pos_callback(self, msg):
        self.hider_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        
        if self.check_gameover():
            self.endgame('Seeker wins')
        
        self.hider_yaw = get_yaw(msg.pose.pose.orientation)
        angle = calc_angle_robots(self.hider_pos,self.hider_yaw,self.seeker_pos)

        if can_see(angle, self.hider_pos, self.seeker_pos):
            print("HIDER CAN SEE SEEKER")
            self.publish_str_msg(self.hider_pub,"Angle " + str(angle))

    def seeker_pos_callback(self, msg):
        self.seeker_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        
        if self.check_gameover():
            self.endgame('Seeker wins')

        self.seeker_yaw = get_yaw(msg.pose.pose.orientation)
        angle = calc_angle_robots(self.seeker_pos,self.seeker_yaw,self.hider_pos)

        if can_see(angle, self.seeker_pos, self.hider_pos):
            print("SEEKER CAN SEE HIDER")
            self.publish_str_msg(self.seeker_pub,"Angle " + str(angle))

    def check_gameover(self):
        return False #temporary
        if not(self.hider_started and self.seeker_started):
            return False

        distance = sqrt(((self.hider_pos[0] - self.seeker_pos[0]) ** 2) + \
                    ((self.hider_pos[1] - self.seeker_pos[1]) ** 2) \
                    + ((self.hider_pos[2] - self.seeker_pos[2]) ** 2))

        if distance <= DISTANCE_ENDGAME:
            return True

        return False

        

    def endgame(self, msg='Game Over'):
        print(msg)
        self.publish_str_msg(self.hider_pub,GAMEOVER_MSG)
        self.publish_str_msg(self.seeker_pub,GAMEOVER_MSG)
        exit(0)


def main(args=None):
    rclpy.init(args=args)

    hide_seek = HideSeek()

    rclpy.spin(hide_seek)

    hide_seek.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
