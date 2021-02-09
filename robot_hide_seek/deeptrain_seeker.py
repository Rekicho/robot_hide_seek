'''
Adapted from https://github.com/vmayoral/basic_reinforcement_learning
 
@author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''

import gym
import random
import numpy as np
import math
import csv

from robot_hide_seek import seeker_env, deepqlearn
from robot_hide_seek.utils import *

env = gym.make('seekerEnv-v0')

epochs = 100000000
updateTargetNetwork = 10
explorationRate = 1
minibatch_size = 1
learnStart = 0
learningRate = 0.00025
discountFactor = 0.99
memorySize = 100000

deepQ = deepqlearn.DeepQ(11, 5, memorySize, discountFactor, learningRate, learnStart, './training_results/seeker')
deepQ.initNetworks([300,300])

def saveScores(scores):
    csv_columns = ['epoch','average_reward','final_reward']
    try:
        with open('seekers.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in scores:
                writer.writerow(data)
    except IOError:
        pass

def round_observation(observation):
    res = []

    for sensor in observation[0]:
        res.append(round(sensor, 2))

    for _ in range(len(observation[0]), 8):
        res.append(math.inf)

    res.append(round(observation[1], 2))
    res.append(round(observation[2], 2))
    res.append(observation[3])

    return np.array(res)

scores = []
stepCounter = 0

# number of reruns
for epoch in range(epochs):
    observations = env.reset()

    for i, observation in enumerate(observations):
        observations[i] = round_observation(observation)

    current_seeker = 0
    done = False

    sum_reward = 0.0
    t = 0
    while not done:
        qValues = deepQ.getQValues(observations[current_seeker])

        action = deepQ.selectAction(qValues, explorationRate)

        newObservation, reward, done, info = env.step(action)

        newObservation = round_observation(newObservation)

        deepQ.addMemory(observations[current_seeker], action, reward, newObservation, done)

        if stepCounter >= learnStart:
            if stepCounter <= updateTargetNetwork:
                deepQ.learnOnMiniBatch(minibatch_size, False)
            else:
                deepQ.learnOnMiniBatch(minibatch_size, True)

        observations[current_seeker] = newObservation
        current_seeker = (current_seeker + 1) % N_SEEKERS
        t += 1
        sum_reward += reward

        if done:
            deepQ.saveModel()
            sum_reward -= reward

            if t != 1:
                scores.append({'epoch': epoch, 'average_reward': sum_reward/(t-1), 'final_reward': reward})
                print("Episode " + str(epoch) + " finished after {} timesteps".format(t) + ". Average Reward: " + str(sum_reward/(t-1)))

            else:
                scores.append({'epoch': epoch, 'average_reward': sum_reward/t, 'final_reward': reward})
                print("Episode " + str(epoch) + " finished after {} timesteps".format(t) + ". Average Reward: " + str(sum_reward/t))

            saveScores(scores)
            break

        stepCounter += 1
        if stepCounter % updateTargetNetwork == 0:
            deepQ.updateTargetNetwork()

    explorationRate *= 0.999
    explorationRate = max(0.05, explorationRate)