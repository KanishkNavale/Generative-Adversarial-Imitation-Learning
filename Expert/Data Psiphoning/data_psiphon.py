import gym 
import sys
import numpy as np
sys.path.insert(1,'Expert/Training')
from DDQN import Agent

# Deconstruct Environment
env = gym.make('CartPole-v1')

# Initiate the Agent and load the weights
agent = Agent(lr=0.0005, gamma=0.99, n_actions= env.action_space.n, epsilon=1.0, batch_size=64, input_dims= env.observation_space.shape, save_agent=False)
for i in range(10):
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
# Load the pretrained weights
agent.q_eval.load_weights('Expert/Training/data/q_eval.h5')
agent.q_next.load_weights('Expert/Training/data/q_next.h5')
print (f'Model Loaded with Trained Weights!')

# Log the data
dataframe = []
for i in range(int(1e2)):
    done = False
    # Initial Reset of Environment
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action) 
        dataframe.append(np.array([observation, action], dtype=object))
        observation = observation_ 
    print(f'Capturing the Data from the Episode: {i}')

# Save the dataset
np.savez('Expert/Data Psiphoning/data/expert_dataset', np.vstack(dataframe), allow_pickle=False)