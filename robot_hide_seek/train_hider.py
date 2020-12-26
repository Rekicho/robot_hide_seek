import gym
from gym import wrappers

import numpy as np
import time

from robot_hide_seek import hider_env, qlearn
from robot_hide_seek.utils import *

from functools import reduce

def round_observation(observation):
    sensors = []

    for sensor in observation[0]:
        sensors.append(str(round(sensor, 2)))

    observation[0] = sensors

    observation[1] = str(round(observation[1], 2))
    observation[2] = str(round(observation[2], 2))

    return observation

# Assumes gazebo simulation and all other robots already running
def main(_args=None):
    env = gym.make('hiderEnv-v0')
    outdir = './training_results'

    env = wrappers.Monitor(env, outdir, force=True) #Force deletes all past training results

    qlearn_alg = qlearn.QLearn(actions=range(env.action_space.n),
                alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
    initial_epsilon = qlearn_alg.epsilon

    start_time = time.time()
    highest_reward = 0

    for x in range(NEPISODES):
        print('Strating Episode #' + str(x))

        cumulated_reward = 0
        done = False

        if qlearn_alg.epsilon > 0.05:
            qlearn_alg.epsilon *= EPSILON_DISCOUNT

        observation = env.reset()
        state = ''.join(map(str, round_observation(observation)))

        while True:
            action = qlearn_alg.chooseAction(state)

            # Execute the action in the environment and get feedback
            observation, reward, done, _info = env.step(action)
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, round_observation(observation)))

            qlearn_alg.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
                print("DONE")
                break

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn_alg.alpha,2))+" - gamma: "+str(round(qlearn_alg.gamma,2))+" - epsilon: "+str(round(qlearn_alg.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))
    
    print( ("\n|"+str(NEPISODES)+"|"+str(qlearn_alg.alpha)+"|"+str(qlearn_alg.gamma)+"|"+str(initial_epsilon)+"*"+str(EPSILON_DISCOUNT)+"|"+str(highest_reward)+"| PICTURE |"))

    env.close()

if __name__ == '__main__':
    main()
