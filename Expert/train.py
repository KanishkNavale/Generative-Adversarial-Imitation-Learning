# Library Imports
import gym
import numpy as np
from DDPG import Agent
import os

data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0')

    n_games = 1000
    score_history = []
    mean_history = []
    best_score = env.reward_range[0]
    mean_score = 0

    # Initiate Training
    agent = Agent(env, data_path, n_games, noise='normal')

    for i in range(n_games):
        score = 0
        done = False

        # Initial Reset of Environment
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            # Store the experience
            agent.store(observation, action, reward, observation_, done)

            observation = observation_
            score += reward

        # Agent Optimize
        agent.optimize()

        score_history.append(score)
        mean_score = np.mean(score_history[-100:])
        mean_history.append(mean_score)

        if mean_score > best_score:
            best_score = mean_score
            agent.save_models()
            print(
                f'Episode:{i}'
                f'\tACC. Rewards: {score:3.4f}'
                f'\tAVG. Rewards: {mean_score:3.4f}'
                f'\t *** MODEL SAVED! ***')
        else:
            print(
                f'Episode:{i}'
                f'\tACC. Rewards: {score:3.4f}'
                f'\tAVG. Rewards: {mean_score:3.4f}')

        # Save the Training data
        np.save(data_path + 'score_history', score_history, allow_pickle=False)
        np.save(data_path + 'avg_history', mean_history, allow_pickle=False)
