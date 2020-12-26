import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry

import math
from functools import partial

from robot_hide_seek.utils import *

class HideSeek(Node):
    
    def __init__(self):
        super().__init__('hide_seek')

        self.declare_parameter('n_hiders')
        self.n_hiders = self.get_parameter('n_hiders').value
        self.declare_parameter('n_seekers')
        self.n_seekers = self.get_parameter('n_seekers').value
        
        self.hider_started = False
        self.seeker_started = False

        self.hider_pos = [[0, 0, 0] for i in range(self.n_hiders)]
        self.seeker_pos = [[0, 0, 0] for i in range(self.n_seekers)]
        self.hider_yaw = [0 for i in range(self.n_hiders)]
        self.seeker_yaw = [[0, 0, 0] for i in range(self.n_seekers)]

        self.time = -1

        self.clock_sub = self.create_subscription(
            Clock, 
            '/clock', 
            self.clock_callback,
            10)

        self.hider_pub = [self.create_publisher(
            String,
            '/hider_' + str(i) + '/game',
            10
        ) for i in range(self.n_hiders)]

        self.seeker_pub = [self.create_publisher(
            String,
            '/seeker_' + str(i) + '/game',
            10
        ) for i in range(self.n_seekers)]

        self.hider_pos_sub = [self.create_subscription(
            Odometry,
            '/hider_' + str(i) + '/odom',
            partial(self.hider_pos_callback, i),
            10
        ) for i in range(self.n_hiders)]

        self.seeker_pos_sub = [self.create_subscription(
            Odometry,
            '/seeker_' + str(i) + '/odom',
            partial(self.seeker_pos_callback, i),
            10
        ) for i in range(self.n_seekers)]

    def clock_callback(self, msg):
        if int(msg.clock.sec) < self.time:
            self.hider_started = False
            self.seeker_started = False
            self.hider_pos = [[0, 0, 0] for i in range(self.n_hiders)]
            self.seeker_pos = [[0, 0, 0] for i in range(self.n_seekers)]
            self.hider_yaw = [0 for i in range(self.n_hiders)]
            self.seeker_yaw = [[0, 0, 0] for i in range(self.n_seekers)]

        self.time = int(msg.clock.sec)

        if self.time == SECONDS_HIDER_START and not self.hider_started:
            self.hider_started = True

        if self.time == SECONDS_SEEKER_START and not self.seeker_started:
            self.seeker_started = True

        if self.time == GAME_TIME_LIMIT:
            self.endgame('Hider wins')

    def publish_str_msg(self, publisher, msg_data):
        msg = String()
        msg.data = msg_data
        publisher.publish(msg)

    # def start_hider(self):
    #     self.hider_started = True

    #     for pub in self.hider_pub:
    #         self.publish_str_msg(pub,START_MSG)

    # def start_seeker(self):
    #     self.seeker_started = True

    #     for pub in self.seeker_pub:
    #         self.publish_str_msg(pub,START_MSG)

    def hider_pos_callback(self, id, msg):
        self.hider_pos[id] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.hider_yaw[id] = get_yaw(msg.pose.pose.orientation)

        if self.check_gameover():
            self.endgame('Seeker wins')

        message = 'POSITIONS\n\n'

        for i in range(self.n_seekers):
            angle = calc_angle_robots(self.hider_pos[id], self.hider_yaw[id], self.seeker_pos[i])
            
            if can_see(angle, self.hider_pos[id], self.seeker_pos[i]):
                distance = calc_distance(self.hider_pos[id], self.seeker_pos[i])

                message += 'Angle: ' + str(angle) + '\nDistance: ' + str(distance) + '\n\n'
                
            else:
                message += 'Angle: ' + str(math.inf) + '\nDistance: ' + str(math.inf) + '\n\n'
            
        self.publish_str_msg(self.hider_pub[id], message)

    def seeker_pos_callback(self, id, msg):
        self.seeker_pos[id] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.seeker_yaw[id] = get_yaw(msg.pose.pose.orientation)

        if self.check_gameover():
            self.endgame('Seeker wins')

        message = POSITIONS_MSG_HEADER + '\n\n'

        for i in range(self.n_hiders):
            angle = calc_angle_robots(self.seeker_pos[id], self.seeker_yaw[id], self.hider_pos[i])
            
            if can_see(angle, self.seeker_pos[id], self.hider_pos[i]):
                distance = calc_distance(self.seeker_pos[id], self.hider_pos[i])

                message += 'Angle: ' + str(angle) + '\nDistance: ' + str(distance) + '\n\n'
                
            else:
                message += 'Angle: ' + str(math.inf) + '\nDistance: ' + str(math.inf) + '\n\n'
            
        self.publish_str_msg(self.seeker_pub[id], message)

    def check_gameover(self):
        if not(self.hider_started and self.seeker_started):
            return False

        gameover = False

        for hider_pos in self.hider_pos:
            found = False

            for seeker_pos in self.seeker_pos:
                distance = calc_distance(hider_pos, seeker_pos)

                if distance <= DISTANCE_ENDGAME:
                    found = True
                    break
            
            if found:
                gameover = True
                break

        return gameover

    def endgame(self, msg='Game Over'):
        for pub in self.hider_pub:
            self.publish_str_msg(pub,GAMEOVER_MSG)

        for pub in self.seeker_pub:
            self.publish_str_msg(pub,GAMEOVER_MSG)


def main(args=None):
    rclpy.init(args=args)

    hide_seek = HideSeek()

    rclpy.spin(hide_seek)

    hide_seek.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
