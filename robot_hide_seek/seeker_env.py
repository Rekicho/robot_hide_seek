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

from robot_hide_seek import seeker_train, gazebo_connection, hider, game_controller
from robot_hide_seek.utils import *

reg = register(
    id='seekerEnv-v0',
    entry_point='robot_hide_seek.seeker_env:SeekerEnv',
    # timestep_limit=100,
)

class SeekerEnv(gym.Env):
    def __init__(self):
        rclpy.init()

        self.hiders = [hider.Hider(0),hider.Hider(1)] # Train basic hider (change to hider_train to train with already hider trained)
        self.seekers = [seeker_train.SeekerTrain(0),seeker_train.SeekerTrain(1)]
        self.game_controller = game_controller.HideSeek()

        self.hider_threads = [threading.Thread(target=self.run_hider, args=(0,), daemon=True), threading.Thread(target=self.run_hider, args=(1,), daemon=True)]
        self.seeker_threads = [threading.Thread(target=self.run_seeker, args=(0,), daemon=True), threading.Thread(target=self.run_seeker, args=(1,), daemon=True)]
        self.game_controller_thread = [threading.Thread(target=self.run_game_controller, daemon=True), threading.Thread(target=self.run_game_controller, daemon=True)]
        
        for thread in self.hider_threads:
            thread.start()

        self.gazebo = gazebo_connection.GazeboConnection(self.game_controller)
        self.action_space = spaces.Discrete(5)
        self.reward_range = (-math.inf, math.inf)

        self.current_seeker = 0

        self.seeker = seeker_train.SeekerTrain(0)

        self._seed()

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_seeker = 0

        for hider in self.hiders:
            hider.reset()
        for seeker in self.seekers:
            seeker.reset()
        self.game_controller.reset()

        self.gazebo.resetSim()
        self.gazebo.unpauseSim()

        time.sleep(SECONDS_SEEKER_START)

        observations = []
        observations.append(self.take_observation())
        self.current_seeker = (self.current_seeker + 1) % len(self.hiders)
        observations.append(self.take_observation())
        self.current_seeker = (self.current_seeker + 1) % len(self.hiders)

        self.gazebo.pauseSim()

        return observations

    def step(self, action):
        seeker = self.seekers[self.current_seeker]
        
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
            seeker.vel_pub
        except AttributeError:
            pass
        else:
            seeker.vel_pub.publish(vel)

        time.sleep(RUNNING_STEP / len(self.seekers))
        observation = self.take_observation()
        self.gazebo.pauseSim()

        reward, done = self.process_observation(observation) #Probably take into consideration distance, angle and time left

        self.current_seeker = (self.current_seeker + 1) % len(self.seekers)

        return observation, reward, done, {}

    def take_observation(self):
        sensors = self.seekers[self.current_seeker].lidar_sensors[:]        

        return [sensors, self.seekers[self.current_seeker].follow_angle, self.seekers[self.current_seeker].follow_distance, self.seekers[self.current_seeker].time, self.seekers[self.current_seeker].result]

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

            reward += (1 - observation[3] / GAME_TIME_LIMIT) * TIME_REWARD # inverse of hider (1-x)*k

        return reward, done

    def run_hider(self, id):
        rclpy.spin(self.hiders[id])
        self.hiders[id].destroy_node()
        rclpy.shutdown()

    def run_seeker(self, id):
        rclpy.spin(self.seekers[id])
        self.seekers[id].destroy_node()
        rclpy.shutdown()

    def run_game_controller(self, id):
        rclpy.spin(self.game_controller)
        self.game_controller.destroy_node()
        rclpy.shutdown()
