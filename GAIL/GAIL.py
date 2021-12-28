from email import policy
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

abs_path = os.getcwd()


class Actor(keras.Model):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, n_actions, density=1024, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name + '.h5'

        self.H1 = tf.keras.layers.Dense(density, activation='relu')
        self.H2 = tf.keras.layers.Dense(density, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.1)
        self.H3 = tf.keras.layers.Dense(density, activation='relu')
        self.H4 = tf.keras.layers.Dense(density, activation='relu')
        self.mu = tf.keras.layers.Dense(n_actions, activation='softmax')

    @tf.function()
    def call(self, state):
        state = self.H1(state)
        state = self.H2(state)
        state = self.drop(state)
        state = self.H3(state)
        state = self.H4(state)
        mu = self.mu(state)
        return mu


class Discriminator(keras.Model):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, density=512, name='critic'):
        super(Discriminator, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name + '.h5'

        self.H1 = tf.keras.layers.Dense(density, activation='relu')
        self.H2 = tf.keras.layers.Dense(density, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.1)
        self.H3 = tf.keras.layers.Dense(density, activation='relu')
        self.H4 = tf.keras.layers.Dense(density, activation='relu')
        self.Q = tf.keras.layers.Dense(1)

    @tf.function()
    def call(self, state, action):
        state = tf.cast(state, tf.float64)
        action = tf.cast(action, tf.float64)
        action = self.H1(tf.concat([state, action], axis=1))
        action = self.H2(action)
        action = self.drop(action)
        action = self.H3(action)
        action = self.H4(action)
        Q = self.Q(action)
        return Q


class Buffer():
    def __init__(self, max_size, state_shape, action_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros(
            (self.mem_size, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros(
            (self.mem_size, *action_shape), dtype=np.float32)

    def store_transition(self, state, action):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action

        self.mem_cntr += 1

    def memorize_expert(self, state, action):
        self.state_memory = state
        self.action_memory = action
        self.mem_cntr = len(state)

    def sample_buffer(self, batch_size):
        max_mem = 2500
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]

        return states, actions


class Agent:
    def __init__(self, expert_obs, expert_actions, env, batch_size):
        self.env = env

        # Action Space Augmentation
        expert_actions = expert_actions.tolist()
        for i in range(len(expert_actions)):
            if expert_actions[i] == 0:
                expert_actions[i] = np.array([1, 0], dtype=np.float32)
            elif expert_actions[i] == 1:
                expert_actions[i] = np.array([0, 1], dtype=np.float32)
        expert_actions = np.vstack(expert_actions)

        self.batch_size = batch_size
        self.exp_obs = expert_obs
        self.exp_actions = expert_actions
        self.mem_size = len(self.exp_obs)
        self.expert_memory = Buffer(
            self.mem_size, self.exp_obs[0].shape, self.exp_actions[0].shape)
        self.replay_memory = Buffer(
            self.mem_size, self.exp_obs[0].shape, self.exp_actions[0].shape)
        n_actions = env.action_space.n

        self.actor = Actor(n_actions)
        self.discriminator = Discriminator()
        self.actor.compile(optimizer=Adam(1e-3))
        self.discriminator.compile(optimizer=Adam(2e-3))

        self.d_loss = []
        self.a_loss = []

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)

    def choose_action(self, observation):
        state = np.array([observation])
        actions = self.actor(state)
        return actions

    def memorize_expert(self):
        print(f'Memorizing Expert Data')
        self.expert_memory.memorize_expert(self.exp_obs, self.exp_actions)
        print('Done!')

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimize(self, steps):

        for i in range(steps):
            # Sample Expert datasets & Policy Dataset
            exp_states, exp_actions = self.expert_memory.sample_buffer(
                self.batch_size)
            states = exp_states

            # Optimize the Descriminator and Actor
            with tf.GradientTape() as desc_tape, tf.GradientTape() as actor_tape:
                actions = tf.squeeze(self.choose_action(states))

                expert_probs = self.discriminator(exp_states, exp_actions)
                policy_probs = self.discriminator(states, actions)

                # Loss Functions implemented as per the GAIL research paper
                # Optimize the Descriminator
                desc_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=policy_probs, labels=tf.ones_like(policy_probs)) + tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=expert_probs, labels=tf.zeros_like(expert_probs))
                desc_gradients = desc_tape.gradient(
                    desc_loss, self.discriminator.trainable_weights)
                self.discriminator.optimizer.apply_gradients(
                    zip(desc_gradients, self.discriminator.trainable_weights))

                # Optimize the Actor
                Q = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=policy_probs, labels=tf.zeros_like(policy_probs))
                actor_loss = tf.math.reduce_mean(tf.multiply(actions, Q)) - (1e-3 * -tf.math.reduce_sum(
                    tf.multiply(actions, tf.math.log(actions + 1e-11))/tf.math.reduce_sum(actions)))
                actor_gradients = actor_tape.gradient(
                    actor_loss, self.actor.trainable_weights)
                self.actor.optimizer.apply_gradients(
                    zip(actor_gradients, self.actor.trainable_weights))

        print(
            f'\tDesc. Loss: {np.mean(self.d_loss):0.4f}, Actor Loss: {np.mean(self.a_loss):0.4f}')
        self.d_loss.append(np.mean(desc_loss))
        self.a_loss.append(np.mean(actor_loss))
        np.save(os.getcwd()+'/GAIL/data/d_loss',
                self.d_loss, allow_pickle=False)
        np.save(os.getcwd()+'/GAIL/data/a_loss',
                self.a_loss, allow_pickle=False)

    def save_model(self):
        self.actor.save_weights(abs_path+'/GAIL/data/actor.h5')
        self.discriminator.save_weights(abs_path+'/GAIL/data/discriminator.h5')
