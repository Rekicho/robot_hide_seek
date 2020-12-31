import gym
from gym import wrappers

import numpy as np
import time

from robot_hide_seek import seeker_env, qlearn
from robot_hide_seek.utils import *

def round_observation(observation):
    res = []

    for sensor in observation[0]:
        res.append(str(round(sensor, 2)))

    res.append(str(round(observation[1], 2)))
    res.append(str(round(observation[2], 2)))
    res.append(observation[3])

    return res

# Assumes gazebo simulation already running
def main(_args=None):
    env = gym.make('seekerEnv-v0')

    qlearn_alg = qlearn.QLearn(actions=range(env.action_space.n),
                alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, res_path='./training_results/seekers.txt')
    initial_epsilon = qlearn_alg.epsilon

    start_time = time.time()
    highest_reward = 0

    for x in range(NEPISODES):
        print('Starting Episode #' + str(x))

        cumulated_reward = 0
        current_seeker = 0
        done = False

        if qlearn_alg.epsilon > 0.05:
            qlearn_alg.epsilon *= EPSILON_DISCOUNT

        observations = env.reset()
        states = [','.join(map(str, round_observation(observations[0]))), ','.join(map(str, round_observation(observations[1])))]

        while True:
            state = states[current_seeker]
            action = qlearn_alg.chooseAction(state)

            # Execute the action in the environment and get feedback
            observation, reward, done, _info = env.step(action)
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ','.join(map(str, round_observation(observation)))

            qlearn_alg.learn(state, action, reward, nextState)

            if not(done):
                states[current_seeker] = nextState
                current_seeker = (current_seeker + 1) % N_SEEKERS
            else:
                qlearn_alg.save('./training_results/seekers.txt')
                print("DONE")
                break

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn_alg.alpha,2))+" - gamma: "+str(round(qlearn_alg.gamma,2))+" - epsilon: "+str(round(qlearn_alg.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))
    
    print( ("\n|"+str(NEPISODES)+"|"+str(qlearn_alg.alpha)+"|"+str(qlearn_alg.gamma)+"|"+str(initial_epsilon)+"*"+str(EPSILON_DISCOUNT)+"|"+str(highest_reward)+"| PICTURE |"))

    env.close()

if __name__ == '__main__':
    main()
