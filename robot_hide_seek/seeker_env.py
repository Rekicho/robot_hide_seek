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

from robot_hide_seek import seeker_train, gazebo_connection
from robot_hide_seek.utils import *

reg = register(
    id='seekerEnv-v0',
    entry_point='robot_hide_seek.seeker_env:SeekerEnv',
    # timestep_limit=100,
)

class SeekerEnv(gym.Env):
    def __init__(self):
        rclpy.init()

        self.seeker = seeker_train.SeekerTrain(0)

        self.seeker_thread = threading.Thread(target=self.run_seeker, daemon=True)
        self.seeker_thread.start()

        self.gazebo = gazebo_connection.GazeboConnection(self.seeker)
        self.action_space = spaces.Discrete(5)
        self.reward_range = (-math.inf, math.inf)

        self._seed()

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()

        time.sleep(SECONDS_SEEKER_START)

        observation = self.take_observation()

        self.gazebo.unpauseSim()

        return observation

    def step(self, action):
        vel = Twist()

        if action == 0: #Forward
            vel.linear.x = SEEKER_LINEAR_SPEED
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
            vel.linear.x = -SEEKER_LINEAR_SPEED
            vel.angular.z = 0.0

        self.gazebo.unpauseSim()

        try:
            self.seeker.vel_pub
        except AttributeError:
            pass
        else:
            self.seeker.vel_pub(vel)

        time.sleep(RUNNING_STEP)
        observation = self.take_observation()
        self.gazebo.pauseSim()

        reward, done = self.process_observation(observation) #Probably take into consideration distance, angle and time left

        return observation, reward, done, {}

    def take_observation(self):
        sensors = self.seeker.lidar_sensors[:]        

        return [sensors, self.seeker.follow_angle, self.seeker.follow_distance, self.seeker.time, self.seeker.result]

    def process_observation(self, observation):
        reward = 0
        done = False

        if observation[4] != 0 or self.seeker.time >= GAME_TIME_LIMIT:
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

            reward += (1 - observation[3] / GAME_TIME_LIMIT) * TIME_REWARD # inverse of hider (1-x)*k

        return reward, done

        
    def run_seeker(self):
        rclpy.spin(self.seeker)
        self.seeker.destroy_node()
        rclpy.shutdown()
