import gym 
import sys
import numpy as np
from DQN import Agent
import tensorflow as tf

# Deconstruct Environment
env= gym.make('MountainCar-v0')
init_state = env.reset()

# Initiate the Agent and load the weights
agent = Agent(env)
print(agent.choose_action(init_state))
agent.train_network = tf.keras.models.load_model('Expert/Training/data/expert_weights.h5')
print (f'Model Loaded with Trained Weights!')

# test the Expert
dataframe = []
for i in range(int(1e2)):
    score = 0
    done = False
    # Initial Reset of Environment
    observation = env.reset()
    while not done:
        env.render()
        action = np.argmax(agent.train_network.predict(np.matrix([observation]))[0])
        observation_, reward, done, info = env.step(action) 
        observation = observation_
        score += reward 
    print(f'Tested Expert on Episode: {i} with ACC. Score: {score}')
    dataframe.append(score)
env.close()

# Save the dataset
np.savez('Expert/Data Psiphoning/data/expert_testscore', dataframe, allow_pickle=False)