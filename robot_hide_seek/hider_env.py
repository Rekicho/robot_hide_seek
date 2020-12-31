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

from robot_hide_seek import hider_train, gazebo_connection, seeker, game_controller
from robot_hide_seek.utils import *

reg = register(
    id='hiderEnv-v0',
    entry_point='robot_hide_seek.hider_env:HiderEnv'
)

class HiderEnv(gym.Env):
    def __init__(self):
        rclpy.init()
        self.executor = rclpy.executors.MultiThreadedExecutor(5)

        self.hiders = [hider_train.HiderTrain(0), hider_train.HiderTrain(1)]
        self.seekers = [seeker.Seeker(0), seeker.Seeker(1)]
        self.game_controller = game_controller.HideSeek()

        for hider_node in self.hiders:
            self.executor.add_node(hider_node)
        for seeker_node in self.seekers:
            self.executor.add_node(seeker_node)
        self.executor.add_node(self.game_controller)

        self.executor_thread = threading.Thread(target=self.run_executor, daemon=True)
        self.executor_thread.start()

        self.gazebo = gazebo_connection.GazeboConnection(self.game_controller)
        self.action_space = spaces.Discrete(5)
        self.reward_range = (-math.inf, math.inf)

        self.current_hider = 0

        self._seed()

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_hider = 0

        for hider_node in self.hiders:
            hider_node.result = 0
            hider_node.time = 0
            hider_node.reset()
        for seeker_node in self.seekers:
            seeker_node.gameover = False
            seeker_node.time = 0
            seeker_node.reset()
        self.game_controller.time = 0
        self.game_controller.reset()

        self.gazebo.resetSim()
        self.gazebo.unpauseSim()

        while True:
            if self.game_controller.time >= SECONDS_HIDER_START:
                break

        observations = []
        observations.append(self.take_observation())
        self.current_hider = (self.current_hider + 1) % len(self.hiders)
        observations.append(self.take_observation())
        self.current_hider = (self.current_hider + 1) % len(self.hiders)

        self.gazebo.pauseSim()

        return observations

    def step(self, action):
        hider = self.hiders[self.current_hider]

        vel = Twist()

        if action == 0: #Forward
            vel.linear.x = HIDER_LINEAR_SPEED
            vel.angular.z = 0.0
        elif action == 1: #Rotate left
            vel.linear.x = 0.0
            vel.angular.z = ROBOT_ANGULAR_SPEED
        elif action == 2: #Rotate right
            vel.linear.x = 0.0
            vel.angular.z = -ROBOT_ANGULAR_SPEED
        elif action == 3: #Stop
            vel.linear.x = 0.0
            vel.angular.z = 0.0
        elif action == 4: #Back
            vel.linear.x = -HIDER_LINEAR_SPEED
            vel.angular.z = 0.0

        self.gazebo.unpauseSim()

        try:
            hider.vel_pub
        except AttributeError:
            pass
        else:
            hider.vel_pub.publish(vel)

        time.sleep(RUNNING_STEP / len(self.hiders))
        observation = self.take_observation()
        self.gazebo.pauseSim()

        reward, done = self.process_observation(observation)

        self.current_hider = (self.current_hider + 1) % len(self.hiders)

        return observation, reward, done, {}

    def take_observation(self):
        sensors = self.hiders[self.current_hider].lidar_sensors[:]        

        return [sensors, self.hiders[self.current_hider].follow_angle, self.hiders[self.current_hider].follow_distance, self.hiders[self.current_hider].time, self.hiders[self.current_hider].result]

    def process_observation(self, observation):
        reward = 0
        done = False

        if observation[4] != 0 or observation[3] >= GAME_TIME_LIMIT:
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


    def run_executor(self):
        self.executor.spin()
        
        for hider_node in self.hiders:
            hider_node.destroy_node()
        for seeker_node in self.seekers:
            seeker_node.destroy_node()
        self.game_controller.destroy_node()

        rclpy.shutdown()