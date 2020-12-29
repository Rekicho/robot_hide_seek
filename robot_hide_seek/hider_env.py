import rclpy
from geometry_msgs.msg import Twist

import gym
from gym.envs.registration import register
from gym import utils, spaces
from gym.utils import seeding

import numpy as np
import time
import threading
import math

from robot_hide_seek import hider_train, gazebo_connection
from robot_hide_seek.utils import *

reg = register(
    id='hiderEnv-v0',
    entry_point='robot_hide_seek.hider_env:HiderEnv'
)

# Decide if train both hiders or only one

class HiderEnv(gym.Env):
    def __init__(self):
        rclpy.init()

        self.hider = hider_train.HiderTrain(0)

        self.hider_thread = threading.Thread(target=self.run_hider, daemon=True)
        self.hider_thread.start()

        self.gazebo = gazebo_connection.GazeboConnection(self.hider)
        self.action_space = spaces.Discrete(5)
        self.reward_range = (-math.inf, math.inf)

        self._seed()

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()

        time.sleep(SECONDS_HIDER_START)

        observation = self.take_observation()

        self.gazebo.unpauseSim()

        return observation

    def step(self, action):
        vel = Twist()

        if action == 0: #Forward
            vel.linear.x = HIDER_LINEAR_SPEED
            vel.angular.z = 0.0
        elif action == 1: #Rotate left
            vel.linear.x = 0.0
            vel.angular.z = ROBOT_ANGULAR_SPEED #CHECK THIS VEL
        elif action == 2: #Rotate right
            vel.linear.x = 0.0
            vel.angular.z = -ROBOT_ANGULAR_SPEED #CHECK THIS VEL
        elif action == 3: #Stop
            vel.linear.x = 0.0
            vel.angular.z = 0.0
        elif action == 4: #Stop
            vel.linear.x = -HIDER_LINEAR_SPEED
            vel.angular.z = 0.0

        self.gazebo.unpauseSim()

        try:
            self.hider.vel_pub
        except AttributeError:
            pass
        else:
            self.hider.vel_pub.publish(vel)

        time.sleep(RUNNING_STEP)
        observation = self.take_observation()
        self.gazebo.pauseSim()

        reward, done = self.process_observation(observation) #Probably take into consideration distance, angle and time left

        return observation, reward, done, {}

    def take_observation(self):
        sensors = self.hider.lidar_sensors[:]        

        return [sensors, self.hider.follow_angle, self.hider.follow_distance, self.hider.time, self.hider.result]

    def process_observation(self, observation):
        reward = 0
        done = False

        if observation[4] != 0 or self.hider.time >= GAME_TIME_LIMIT:
            done = True

        if observation[4] > 0:
            reward = 10000

        elif observation[4] < 0:
            reward = -10000

        else:
            if observation[2] == math.inf:
                reward = 20

            else:
                reward = observation[2]

            reward += (observation[3] / GAME_TIME_LIMIT) * TIME_REWARD

        return reward, done

        
    def run_hider(self):
        rclpy.spin(self.hider)
        self.hider.destroy_node()
        rclpy.shutdown()