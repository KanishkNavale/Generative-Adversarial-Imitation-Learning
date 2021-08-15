# Library Imports
import numpy as np
import matplotlib.pyplot as plt
import os

abs_path = os.getcwd()
 
# Import the datasets
sum_rewards = np.load(abs_path+'/GAIL/data/score_history.npy')
avg_rewards = np.load(abs_path+'/GAIL/data/avg_history.npy')
expert_rewards = np.load(abs_path+'/GAIL/data/GAIL_scores.npy')
    
# Definition to plots
plt.figure()
plt.plot(sum_rewards, c='green', label='Summed Rewards')
plt.plot(avg_rewards, c='blue', label='Average Rewards')
plt.xlabel('Episodes')
plt.grid(True)
plt.legend(loc='best')
plt.title('GAIL Training')
plt.savefig(abs_path+'/GAIL/data/Training.png')

# Definition to plots
plt.figure()
plt.plot(expert_rewards, c='red', label='GAIL Summed Rewards')
plt.xlabel('Episodes')
plt.grid(True)
plt.legend(loc='best')
plt.title('GAIL Play Performance')
plt.savefig(abs_path+'/GAIL/data/Performance.png')