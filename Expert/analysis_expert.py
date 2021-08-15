# Library Imports
import numpy as np
import matplotlib.pyplot as plt

# Import the datasets
sum_rewards = np.load('Expert/Training/data/score_history.npy')
avg_rewards = np.load('Expert/Training/data/avg_history.npy')
expert_rewards = np.load('Expert/Training/data/expert_scores.npy')

# Definition to plots
plt.figure()
plt.plot(sum_rewards, c='red', label='Summed Rewards')
plt.plot(avg_rewards, c='blue', label='Average Rewards')
plt.xlabel('Episodes')
plt.grid(True)
plt.legend(loc='best')
plt.title('Expert Training')
plt.savefig('Expert/Training/data/Expert Training.png')
plt.show()

# Definition to plots
plt.figure()
plt.plot(expert_rewards, c='green', label='Expert Summed Rewards')
plt.xlabel('Episodes')
plt.grid(True)
plt.legend(loc='best')
plt.title('Expert Play Performance')
plt.savefig('Expert/Training/data/Expert Performance.png')
plt.show()