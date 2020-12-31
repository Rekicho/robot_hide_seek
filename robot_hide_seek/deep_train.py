import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gym

from baselines import deepq
from robot_hide_seek import hider_env    

def main(): #Not working
    env = gym.make('hiderEnv-v0')
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10
    )
    print("Saving model to ./training_results/hiderEnv-v0.pkl")
    act.save("./training_results/hiderEnv-v0.pkl")


if __name__ == '__main__':
    main()