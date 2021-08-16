from email import policy
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

abs_path = os.getcwd()

class Actor(keras.Model):
    """ Actor is the Generator Network"""
    def __init__(self, n_actions):
        super(Actor, self).__init__()
    
        self.l1 = Dense(2048, activation='elu')
        self.l2 = Dense(1024, activation='elu')
        self.l3 = Dense(512, activation='elu')
        self.l4 = Dense(256, activation='elu')
        self.l5 = Dense(64, activation='elu')
        self.l6 = Dense(8, activation='elu')
        self.l7 = Dense(n_actions, activation='softmax')
        self.drop = Dropout(0.1)
    
    @tf.function     
    def call(self, x):
        x = self.l1(x)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        x = self.l3(x)
        x = self.drop(x)
        x = self.l4(x)
        x = self.drop(x)
        x = self.l5(x)
        x = self.drop(x)
        x = self.l6(x)
        x = self.l7(x)
        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.l1 = Dense(256, activation='relu')
        self.l2 = Dense(128, activation='relu')
        self.l3 = Dense(64, activation='relu')
        self.l4 = Dense(32, activation='relu')
        self.l5 = Dense(16, activation='relu')
        self.l6 = Dense(8, activation='relu')
        self.l7 = Dense(1, activation='sigmoid')
        self.drop = Dropout(0.1)
    
    @tf.function  
    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.l1(x)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        x = self.l3(x)
        x = self.drop(x)
        x = self.l4(x)
        x = self.drop(x)
        x = self.l5(x)
        x = self.drop(x)
        x = self.l6(x)
        x = self.l7(x)
        return x

class Buffer():
    def __init__(self, max_size, state_shape, action_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *state_shape) ,dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *action_shape) ,dtype=np.float32)

    def store_transition(self, state, action):
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        
        self.mem_cntr +=1
        
    def memorize_expert(self,state,action):
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
                expert_actions[i] = np.array([1,0],dtype=np.float32)
            elif expert_actions[i] == 1:
                expert_actions[i] = np.array([0,1],dtype=np.float32)
        expert_actions = np.vstack(expert_actions)
        
        self.batch_size = batch_size
        self.exp_obs = expert_obs
        self.exp_actions = expert_actions
        self.mem_size = len(self.exp_obs)
        self.expert_memory = Buffer(self.mem_size, self.exp_obs[0].shape, self.exp_actions[0].shape)
        self.replay_memory = Buffer(self.mem_size, self.exp_obs[0].shape, self.exp_actions[0].shape)
        n_actions = env.action_space.n
        
        self.actor = Actor(n_actions)
        self.discriminator = Discriminator()
        self.actor.compile(optimizer= Adam(1e-4))
        self.discriminator.compile(optimizer= Adam(1e-5))
        
        self.d_loss = []
        self.a_loss = []
       
    def choose_action(self, observation):
        state = np.array([observation])
        actions = self.actor(state)
        return actions
    
    def memorize_expert(self):
        print (f'Memorizing Expert Data')
        self.expert_memory.memorize_expert(self.exp_obs, self.exp_actions)
        print('Done!')
    
    def optimize(self, steps):
        
        for i in range(steps):
            # Sample Expert datasets & Policy Dataset
            exp_states, exp_actions = self.expert_memory.sample_buffer(self.batch_size)
            states, _= self.expert_memory.sample_buffer(self.batch_size)

            # Optimize the Descriminator and Actor
            with tf.GradientTape() as desc_tape, tf.GradientTape() as actor_tape: 
                actions = tf.squeeze(self.choose_action(states))
                
                expert_probs = self.discriminator(exp_states, exp_actions)
                policy_probs = self.discriminator(states, actions)
                
                # Loss Functions implemented as per the GAIL research paper
                # Optimize the Descriminator
                desc_loss = tf.math.log(policy_probs) + tf.math.reduce_mean(tf.math.log(1.0 - expert_probs))   
                desc_gradients = desc_tape.gradient(desc_loss, self.discriminator.trainable_weights)
                self.discriminator.optimizer.apply_gradients(zip(desc_gradients, self.discriminator.trainable_weights))
                
                # Optimize the Actor
                log_policy = tf.math.log(actions)
                entropy = tf.math.reduce_mean(-log_policy)
                Q = tf.reduce_mean(tf.math.log(self.discriminator(states,actions)))
                actor_loss = tf.math.reduce_mean(log_policy * Q) - (1e-3 * entropy)
                actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_weights)
                self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_weights))
                
        print (f'\tDesc. Loss: {np.mean(self.d_loss):0.4f}, Actor Loss: {np.mean(self.a_loss):0.4f}')
        self.d_loss.append(np.mean(desc_loss))
        self.a_loss.append(np.mean(actor_loss))
        np.save(os.getcwd()+'/GAIL/data/d_loss', self.d_loss, allow_pickle=False)
        np.save(os.getcwd()+'/GAIL/data/a_loss', self.a_loss, allow_pickle=False)
 
    def save_model(self):
        self.actor.save_weights(abs_path+'/GAIL/data/actor.h5')
        self.discriminator.save_weights(abs_path+'/GAIL/data/discriminator.h5')
