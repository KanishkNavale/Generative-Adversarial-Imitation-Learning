# Library Imports
import gym
import numpy as np
from GAIL import Agent

# Load the Environment
env = gym.make('CartPole-v1')

# Load the Expert Dataset and Agent
expert_obs = np.load('Expert/data/expert_DatasetStates.npy')
expert_actions = np.load('Expert/data/expert_DatasetAction.npy')
agent = Agent(expert_obs, expert_actions, env, batch_size=64)
agent.memorize_expert()

n_games = 2000
score_history = []
avg_history = []
best_score = env.reward_range[0]
avg_score = 0

for i in range(n_games):
    score = 0
    done = False

    # Initial Reset of Environment
    observation = env.reset()

    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.replay_memory.store_transition(observation, action)
        observation = observation_
        score += reward
        
    agent.optimize(16)
        
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    avg_history.append(avg_score)

    if avg_score > best_score:
        best_score = avg_score
        agent.save_model()
        print(f'Episode:{i} \t ACC. Rewards: {score} \t AVG. Rewards: {avg_score:3.2f} \t *** MODEL SAVED! ***')
    else:
        print(f'Episode:{i} \t ACC. Rewards: {score} \t AVG. Rewards: {avg_score:3.2f}')
        
    # Save the Training data and Model Loss
    np.save('GAIL/data/score_history', score_history, allow_pickle=False)
    np.save('GAIL/data/avg_history', avg_history, allow_pickle=False)
