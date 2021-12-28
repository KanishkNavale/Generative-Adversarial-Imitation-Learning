import gym
import os
import numpy as np
from DDPG import Agent

data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0')
    n_games = 100

    # Load the trained Weights
    agent = Agent(env, data_path, n_games, noise='normal')
    agent.load_models()

    # Test the Trained Agent
    score_log = []
    mean_log = []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            score += reward

        print(f'Expert Tryouts:{i}\tACC. Rewards: {score}')
        score_log.append(score)
        mean_log.append(np.mean(score_log[-5:]))

    # Save the log
    np.save(data_path + 'expert_sum', score_log, allow_pickle=False)
    np.save(data_path + 'expert_mean', score_log, allow_pickle=False)
