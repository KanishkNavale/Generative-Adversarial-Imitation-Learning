import gym 
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
dataframe_actions = []
dataframe_states = []
for i in range(int(2500)):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.expert_action(observation)
        observation_, reward, done, info = env.step(action)
        # Build dataframe
        dataframe_states.append(observation)
        dataframe_actions.append(action)
        observation = observation_
        score += reward
    print(f'Logging Data from Episode:{i} with ACC. Rewards: {score}')

    # Save the dataframe    
    np.save('data/expert_DatasetStates', dataframe_states, allow_pickle=False)
    np.save('data/expert_DatasetAction', dataframe_actions, allow_pickle=False)
