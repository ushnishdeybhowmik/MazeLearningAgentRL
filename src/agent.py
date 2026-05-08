import numpy as np
import random

class QLearningAgent:

    def __init__(self):

        self.q_table = np.zeros((4, 4, 4))

        self.learning_rate = 0.1
        self.discount_factor = 0.9

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def choose_action(self, state):

        row, col = state

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)

        return np.argmax(self.q_table[row, col])

    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state
    ):

        row, col = state
        next_row, next_col = next_state

        current_q = self.q_table[row, col, action]

        max_future_q = np.max(
            self.q_table[next_row, next_col]
        )

        new_q = current_q + self.learning_rate * (
            reward +
            self.discount_factor * max_future_q -
            current_q
        )

        self.q_table[row, col, action] = new_q

    def decay_epsilon(self):

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )