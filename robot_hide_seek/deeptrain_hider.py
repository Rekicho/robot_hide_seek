import gym
import random
import numpy as np
import math

from robot_hide_seek import hider_env, deepqlearn
from robot_hide_seek.utils import *

env = gym.make('hiderEnv-v0')

epochs = 1000000
# steps = 100000
updateTargetNetwork = 10
explorationRate = 1
minibatch_size = 1 #128
learnStart = 0 #128
learningRate = 0.00025
discountFactor = 0.99
memorySize = 100000

last10Scores = [0] * 10
last10ScoresIndex = 0
last10Filled = False

deepQ = deepqlearn.DeepQ(11, 5, memorySize, discountFactor, learningRate, learnStart)
# deepQ.initNetworks([30,30,30])
# deepQ.initNetworks([30,30])
deepQ.initNetworks([300,300])


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


stepCounter = 0

# number of reruns
for epoch in range(epochs):
    observations = env.reset()

    for i, observation in enumerate(observations):
        observations[i] = round_observation(observation)

    current_seeker = 0
    done = False

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

        if done:
            last10Scores[last10ScoresIndex] = t
            last10ScoresIndex += 1
            if last10ScoresIndex >= 10:
                last10Filled = True
                last10ScoresIndex = 0
            if not last10Filled:
                print("Episode " + str(epoch) + " finished after {} timesteps".format(t+1))
            else :
                print("Episode " + str(epoch) + " finished after {} timesteps".format(t+1) + " last 10 average: " + str(sum(last10Scores)/len(last10Scores)))
            break

        stepCounter += 1
        if stepCounter % updateTargetNetwork == 0:
            deepQ.updateTargetNetwork()

    explorationRate *= 0.995
    # explorationRate -= (2.0/epochs)
    explorationRate = max(0.05, explorationRate)