# Library Imports
import numpy as np
import matplotlib.pyplot as plt
import os

data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

# Import the datasets
sum_rewards = np.load(data_path + 'score_history.npy')
avg_rewards = np.load(data_path + 'avg_history.npy')
expert_sum = np.load(data_path + 'expert_sum.npy')
expert_mean = np.load(data_path + 'expert_mean.npy')

# Definition to plots
plt.figure(1)
plt.plot(sum_rewards, alpha=0.3, label='Episode Sum')
plt.plot(avg_rewards, label='Moving Mean')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.grid(True)
plt.legend(loc='best')
plt.title('Expert Training')
plt.savefig(data_path + 'Expert_Training.png')

# Definition to plots
plt.figure(2)
plt.plot(expert_sum, alpha=0.3, label='Episode Sum')
plt.plot(expert_mean, label='Moving Mean')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.grid(True)
plt.legend(loc='best')
plt.title('Expert Play Performance')
plt.savefig(data_path + 'Expert_Performance.png')
