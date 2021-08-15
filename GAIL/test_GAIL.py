import gym 
import numpy as np
import os
import sys
from GAIL import Agent
abs_path = os.getcwd()

# Deconstruct Environment
env= gym.make('CartPole-v1')
init_state = env.reset()

# Initiate the Agent and load the weights
expert_obs = np.load(abs_path+'/Expert/data/expert_DatasetStates.npy')
expert_actions = np.load(abs_path+'/Expert/data/expert_DatasetAction.npy')
agent = Agent(expert_obs, expert_actions, env, batch_size=64)

for i in range(10):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(np.argmax(action))
        agent.replay_memory.store_transition(observation, action)
        observation = observation_

# Load the trained Weights
agent.actor.load_weights(os.getcwd()+'/GAIL/data/actor.h5')
      
# Test the Trained Agent
score_log = []
for i in range(100):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(np.argmax(action))
        agent.replay_memory.store_transition(observation, action)
        observation = observation_
        score += score
    print(f'GAIL Tryouts:{i} \t ACC. Rewards: {score}')
    score_log.append(score)
    
# Save the log
np.save(os.getcwd()+'/GAIL/data/GAIL_scores', score_log, allow_pickle=False)
                

        
