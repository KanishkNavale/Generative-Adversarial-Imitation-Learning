from numpy.lib.function_base import gradient
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
from torch import norm

class Actor(keras.Model):
    """ Actor is the Generator Network"""
    def __init__(self, n_actions):
        super(Actor, self).__init__()
    
        self.l1 = Dense(512, activation='relu')
        self.l2 = Dense(128, activation='relu')
        self.l3 = Dense(32, activation='relu')
        self.l4 = Dense(8, activation='relu')
        self.l5 = Dense(n_actions, activation='softmax')
    
    @tf.function     
    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.l1 = Dense(256, activation='relu')
        self.l2 = Dense(64, activation='relu')
        self.l3 = Dense(16, activation='relu')
        self.l4 = Dense(1, activation='sigmoid')
    
    @tf.function  
    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int)

    def store_transition(self, state, action):
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        
        self.mem_cntr +=1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]

        return states, actions
    
class ExpertBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int)

    def store_transition(self, state, action):
        
        self.state_memory = state
        self.action_memory = action

        self.mem_cntr = len(state)

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]

        return states, actions
    
        
class Agent:
    def __init__(self, expert_obs, expert_actions, env, batch_size):
        self.env = env
        
        self.batch_size = batch_size
        self.exp_obs = expert_obs
        self.exp_actions = expert_actions
        self.mem_size = len(self.exp_obs)
        self.expert_memory = ExpertBuffer(self.mem_size, self.exp_obs[0].shape)
        self.replay_memory = ReplayBuffer(self.mem_size, self.exp_obs[0].shape)
        n_actions = env.action_space.n
        
        self.actor = Actor(n_actions)
        self.discriminator = Discriminator()
        self.actor.compile(optimizer= Adam(1e-4))
        self.discriminator.compile(optimizer= Adam(2e-4))
        
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
       
    def choose_action(self, observation):
        state = np.array([observation])
        actions = self.actor(state)
        action = np.int(tf.math.argmax(actions).numpy()[0])
        return action
    
    def memorize_expert(self):
        self.expert_memory.store_transition(self.exp_obs, self.exp_actions)
        print (f'Memorizing Expert Data')
    
    
    def optimize(self, steps):
        
        self.d_loss = []
        self.a_loss = []
            
        for i in range(steps):
            # Sample Expert datasets
            exp_states, exp_actions = self.expert_memory.sample_buffer(self.batch_size)
            states, actions = self.expert_memory.sample_buffer(self.batch_size)
            
            actions = tf.cast(actions, tf.int32)
            exp_actions = tf.cast(exp_actions, tf.int32)
                    
            actions = tf.one_hot(actions, 2)
            exp_actions = tf.one_hot(exp_actions, 2)

            # Optimize the Descriminator and Actor
            with tf.GradientTape() as desc_tape, tf.GradientTape() as actor_tape: 
                
                actions = tf.cast(actions, tf.float32)
                exp_actions = tf.cast(exp_actions, tf.float32) 

                exp_values = self.discriminator(exp_states, exp_actions)
                noisy_values = self.discriminator(states, actions)
                
                # Optimize the Descriminator
                desc_loss = tf.math.reduce_mean(tf.math.log(noisy_values)) + tf.math.reduce_mean(tf.math.log(1.0 - exp_values))   
                desc_gradients = desc_tape.gradient(-desc_loss, self.discriminator.trainable_weights)
                self.discriminator.optimizer.apply_gradients(zip(desc_gradients, self.discriminator.trainable_weights))
                
                # Optimize the Actor
                new_noisy_values = self.discriminator(states, actions)
                action_probs = self.actor(states)
                actions = tf.cast(tf.argmax(action_probs, axis=1), tf.float32)
                
                entropy = tf.math.reduce_mean(-tf.math.log(actions + 1e-8))
                actor_loss = tf.math.reduce_mean(tf.math.log(action_probs) * tf.math.reduce_mean(tf.math.log(new_noisy_values))) - (1e-3 * entropy)
                actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_weights)
                self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_weights))
                
                self.d_loss.append(desc_loss)
                self.a_loss.append(actor_loss)

        print (f' Desc. Loss: {np.mean(self.d_loss):0.4f}, Actor Loss: {np.mean(self.a_loss):0.4f}')
 
    def save_model(self):
        self.actor.save_weights('GAIL/data/actor.h5')
        self.discriminator.save_weights('GAIL/data/discriminator.h5')
