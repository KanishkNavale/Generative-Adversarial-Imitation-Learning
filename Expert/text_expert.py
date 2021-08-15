import gym 
import os
import numpy as np
from DDQN import Agent

# Deconstruct Environment
env= gym.make('CartPole-v1')
init_state = env.reset()

# Initiate the Agent and load the weights
agent = Agent(lr=0.0005, gamma=0.99, n_actions= env.action_space.n, epsilon=1.0, batch_size=64, input_dims= env.observation_space.shape)
for i in range(10):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_

# Load the trained Weights
agent.load_model()

# Test the Trained Agent
score_log = []
for i in range(100):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.expert_action(observation)
        observation_, reward, done, info = env.step(action)
        observation = observation_
        score += reward
    print(f'Expert Tryouts:{i} \t ACC. Rewards: {score}')
    score_log.append(score)

# Save the log
#np.save(os.getcwd()+'Expert/data/expert_scores', score_log, allow_pickle=False)
