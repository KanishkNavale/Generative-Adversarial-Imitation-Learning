import os
import numpy as np
import json

import torch

from sklearn.decomposition import PCA

import gym

import matplotlib.pyplot as plt

from DDQN import Agent


def predict_value(agent: Agent, state: np.ndarray) -> float:
    state = torch.as_tensor(state, dtype=torch.float32)
    value = agent.online_network(state).detach().numpy()
    return np.max(value)


if __name__ == "__main__":

    # Init. path
    data_path = os.path.abspath('Expert/data')

    # Init. Environment and agent
    env = gym.make('CartPole-v1')
    env.reset()

    agent = Agent(env=env, training=False)
    agent.load_models(data_path)
    agent.online_network.to(torch.device("cpu"))

    with open(os.path.join(data_path, 'training_info.json')) as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'testing_info.json')) as f:
        test_data = json.load(f)

    # Load all the data frames
    score = [data["Epidosic Summed Rewards"] for data in train_data]
    average = [data["Moving Mean of Episodic Rewards"] for data in train_data]
    test = [data["Test Score"] for data in test_data]

    # Process network data
    cart_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=500)
    cart_vel = np.linspace(-1e6, 1e6, num=500)
    pole_pos = np.linspace(env.observation_space.low[2], env.observation_space.high[2], num=500)
    pole_vel = np.linspace(-1e6, 1e6, num=500)

    # Compress the states to 2D
    state = np.vstack((cart_pos, cart_vel, pole_pos, pole_vel)).T
    pca = PCA(n_components=2)
    compressed_states = pca.fit_transform(state)
    assert np.allclose(pca.explained_variance_ratio_[0], 1.0)

    # Prepare data to plot
    x, y = compressed_states[:, 0], compressed_states[:, 1]
    x, y = np.meshgrid(x, y)
    z = np.apply_along_axis(lambda _: predict_value(agent, _), 2, pca.inverse_transform(np.dstack([x, y])))
    z = z[:-1, :-1]
    z_min, z_max = z.min(), z.max()

    # Generate graphs
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    axes[0].plot(score, alpha=0.5, label='Episodic summation')
    axes[0].plot(average, label='Moving mean of 100 episodes')
    axes[0].grid(True)
    axes[0].set_xlabel('Training Episodes')
    axes[0].set_ylabel('Rewards')
    axes[0].legend(loc='best')
    axes[0].set_title('Training Profile')

    axes[1].boxplot(test)
    axes[1].grid(True)
    axes[1].set_xlabel('Test Run')
    axes[1].set_title('Testing Profile')

    axes[2].pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    axes[2].axis([x.min(), x.max(), y.min(), y.max()])
    axes[2].set_xlabel('State: Principal Axes-1 (variance = 1.0)')
    axes[2].set_ylabel('State: Principal Axes-2')
    axes[2].set_title("Value Estimation")
    fig.colorbar(axes[2].pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max))

    fig.tight_layout()
    plt.savefig(os.path.join(data_path, 'DDQN Agent Profiling.png'))
