import numpy as np

class Maze:

    def __init__(self):

        self.grid = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 2]
        ])

        self.start = (0, 0)
        self.goal = (3, 3)

        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):

        row, col = self.agent_pos

        if action == 0:    # UP
            new_row, new_col = row - 1, col

        elif action == 1:  # DOWN
            new_row, new_col = row + 1, col

        elif action == 2:  # LEFT
            new_row, new_col = row, col - 1

        else:              # RIGHT
            new_row, new_col = row, col + 1

        # Boundary checks
        if (
            new_row < 0 or
            new_row >= 4 or
            new_col < 0 or
            new_col >= 4
        ):
            return self.agent_pos, -10, False

        # Wall check
        if self.grid[new_row][new_col] == 1:
            return self.agent_pos, -10, False

        self.agent_pos = (new_row, new_col)

        # Goal check
        if self.grid[new_row][new_col] == 2:
            return self.agent_pos, 100, True

        return self.agent_pos, -1, False