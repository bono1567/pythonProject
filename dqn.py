import random
import numpy as np
import os
import gym
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from collections import deque
import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class MountainCarDQN:
    def __init__(self, env, boost=False):
        self.env = env
        self.gamma = 0.99
        self.epsilon = 0.7
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.replay_buffer = deque(maxlen=20000)
        self.train_network = self.create_network()
        self.episode_num = 400
        self.iteration_num = 201  # max is 200
        self.num_pick_from_buffer = 32
        self.target_network = self.create_network()

        self.boost = boost
        self.train_network.summary()

        weight_path = './dqn_model/best_model.h5'
        if 'best_model.h5' in os.listdir('./dqn_model/'):
            print("weight was loaded into the model.")
            self.train_network.load_weights(weight_path)

        self.target_network.set_weights(self.train_network.get_weights())

    def create_network(self):
        model = models.Sequential()
        state_shape = self.env.observation_space.shape

        model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(48, activation='relu'))
        model.add(layers.Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def get_best_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(self.train_network.predict(state)[0])
        return action

    def train_from_buffer(self):
        if self.boost or len(self.replay_buffer) < self.num_pick_from_buffer:
            return

        samples = random.sample(self.replay_buffer, self.num_pick_from_buffer)

        states = []
        new_states = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            new_states.append(new_state)

        states = np.array(states).reshape(self.num_pick_from_buffer, 2)
        targets = self.train_network.predict(states)

        new_states = np.array(new_states).reshape(self.num_pick_from_buffer, 2)
        new_targets = self.target_network.predict(new_states)

        index = 0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[index]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_targets[index])
                target[action] = reward + Q_future * self.gamma
            index += 1
        self.train_network.fit(states, targets, epochs=1, verbose=0)

    def original_try(self, current_state, episode):
        reward_sum = 0

        for i in range(self.iteration_num):
            curr_action = self.get_best_action(current_state)

            if episode % 50 == 0:
                self.env.render()

            new_state, reward, done, _ = self.env.step(curr_action)
            new_state = new_state.reshape(1, 2)
            if new_state[0][0] >= 0.5:
                reward += 10

            self.replay_buffer.append([current_state, curr_action, reward, new_state, done])
            self.train_from_buffer()
            reward_sum += reward
            current_state = new_state

            if done:
                if i >= 199:
                    print("Failed to complete task. Ep:{}".format(episode))
                else:
                    print("Completed Task. Ep:{}".format(episode))
                    self.train_network.save('./dqn_model/best_model.h5'.format(episode))
                break

        self.target_network.set_weights(self.train_network.get_weights())
        self.epsilon -= self.epsilon_decay

    def start(self):
        for eps in range(self.episode_num):
            start_time = datetime.datetime.now()
            current_state = self.env.reset().reshape(1, 2)
            self.original_try(current_state, eps)
            print("Time taken for ep:{} {}".format(eps, datetime.datetime.now() - start_time))

    def simulate(self):
        weight_path = './dqn_model/best_model.h5'
        simulation_model = self.create_network()
        simulation_model.load_weights(weight_path)
        curr_state = self.env.reset().reshape(1, 2)
        done = False
        while not done:
            self.env.render()
            curr_action = np.argmax(simulation_model.predict(curr_state)[0])
            curr_state, _, done, _ = self.env.step(curr_action)
            curr_state = np.array(curr_state).reshape(1, 2)


if __name__ == '__main__':
    environment = gym.make('MountainCar-v0')
    environment.reset()
    model_car = MountainCarDQN(environment)
    model_car.simulate()
