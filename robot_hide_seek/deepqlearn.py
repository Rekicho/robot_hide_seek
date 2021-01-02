'''
Adapted from https://github.com/vmayoral/basic_reinforcement_learning
 
@author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''

import gym
import random
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

from robot_hide_seek import seeker_env
from robot_hide_seek.utils import *

# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# import theano

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists 
    that get returned as another list of dictionaries with each key corresponding to either 
    "state", "action", "reward", "nextState" or "isFinal".
    """
    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getMiniBatch(self, size) :
        indices = random.sample(set(np.arange(len(self.states))), min(size,len(self.states)))
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]})
        return miniBatch

    def getCurrentSize(self) :
        return len(self.states)

    def getMemory(self, index): 
        return {'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]}

    def addMemory(self, state, action, reward, newState, isFinal) :
        if (self.currentPosition >= self.size - 1) :
            self.currentPosition = 0
        if (len(self.states) > self.size) :
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        else :
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)
        
        self.currentPosition += 1

class DeepQ:
    """
    DQN abstraction.
    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')
    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate
   
    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        bias = True
        dropout = 0
        regularizationFactor = 0.01
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else :
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', W_regularizer=l2(regularizationFactor),  bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', bias=bias))

            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(Dense(layerSize, init='lecun_uniform', W_regularizer=l2(regularizationFactor), bias=bias))
                else:
                    model.add(Dense(layerSize, init='lecun_uniform', bias=bias))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(self.output_size, init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,)))#, init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,)))#, init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize))#, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size))#, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("layer " + str(i) + ": " + str(weights))
            i += 1


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        state = np.array(state)
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getTargetQValues(self, state):
        state = np.array(state)
        predicted = self.targetModel.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples        
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), epochs=1, verbose=0)

env = gym.make('seekerEnv-v0')

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

deepQ = DeepQ(11, 5, memorySize, discountFactor, learningRate, learnStart)
# deepQ.initNetworks([30,30,30])
# deepQ.initNetworks([30,30])
deepQ.initNetworks([300,300])


def round_observation(observation):
    res = []

    for sensor in observation[0]:
        res.append(round(sensor, 2))

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

        print(t)

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
            print("updating target network")

    explorationRate *= 0.995
    # explorationRate -= (2.0/epochs)
    explorationRate = max(0.05, explorationRate)