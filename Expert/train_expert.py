# Library Imports
import gym 
import numpy as np
from DDQN import Agent

# Deconstruct Environment
env = gym.make('CartPole-v1')
print (f'Environment Shape: {env.observation_space.shape}')
print (f'Action Shape: {env.action_space.n}')
print (f'Reward Range: {env.reward_range}')
print (f'Max. Episode Steps: {env._max_episode_steps}')

# Initiate Training
agent = Agent(lr=0.0005, gamma=0.99, n_actions= env.action_space.n, epsilon=1.0, batch_size=64, input_dims= env.observation_space.shape)
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
        
        # Optimize the Agent
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        
        observation = observation_
        score += reward
        
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    avg_history.append(avg_score)

    if avg_score > best_score:
        best_score = avg_score
        agent.save_model()
        print(f'Episode:{i} \t ACC. Rewards: {score} \t AVG. Rewards: {avg_score:3.2f} \t *** MODEL SAVED! ***')
    else:
        print(f'Episode:{i} \t ACC. Rewards: {score} \t AVG. Rewards: {avg_score:3.2f}')
        
    # Save the Training data
    np.save('Expert/Training/data/score_history', score_history, allow_pickle=False)
    np.save('Expert/Training/data/avg_history', avg_history, allow_pickle=False)