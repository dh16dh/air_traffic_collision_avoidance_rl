import numpy as np
import random

from gymnasium import spaces


class QLearningAgent:
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Box, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min

        # Initialise Q Table
        self.q_table = np.zeros(self._get_q_table_shape())

    def _get_q_table_shape(self):
        return (self.observation_space['position'].shape[0] + 1,
                self.observation_space['heading'].shape[0] + 1,
                self.action_space.shape[0])

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate + td_error

        if done:
            self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
