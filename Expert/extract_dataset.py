import gym
import numpy as np
from DDPG import Agent
import os

data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

if __name__ == '__main__':
    # Deconstruct Environment
    env = gym.make('MountainCarContinuous-v0')
    n_games = 2500

    # Load the trained Weights
    agent = Agent(env, data_path, n_games, noise='normal')
    agent.load_models()

    # Test the Trained Agent
    actions = []
    states = []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            score += reward

            actions.append(action)
            states.append(observation)

        print(f'Logging Data from Episode:{i}\tACC. Rewards: {score:3.4f}')

        # Save the dataframe
        np.save(data_path + 'ExpStates', states, allow_pickle=False)
        np.save(data_path + 'ExpActions', actions, allow_pickle=False)
