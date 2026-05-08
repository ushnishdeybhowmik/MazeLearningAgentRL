import pygame
import time

from maze import Maze
from agent import QLearningAgent

# -----------------------------
# CONFIG
# -----------------------------

CELL_SIZE = 100
ROWS = 4
COLS = 4

WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE

FPS = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (50, 100, 255)

# -----------------------------
# TRAIN AGENT
# -----------------------------

env = Maze()
agent = QLearningAgent()

episodes = 1000

for episode in range(episodes):

    state = env.reset()
    done = False

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.update_q_table(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

    agent.decay_epsilon()

print("Training Complete")

# -----------------------------
# PYGAME SETUP
# -----------------------------

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Solver RL Agent")

clock = pygame.time.Clock()

# -----------------------------
# DRAW GRID
# -----------------------------

def draw_grid(agent_pos):

    screen.fill(WHITE)

    for row in range(ROWS):
        for col in range(COLS):

            rect = pygame.Rect(
                col * CELL_SIZE,
                row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            cell_value = env.grid[row][col]

            # Empty cell
            color = WHITE

            # Wall
            if cell_value == 1:
                color = BLACK

            # Goal
            elif cell_value == 2:
                color = GREEN

            pygame.draw.rect(screen, color, rect)

            pygame.draw.rect(screen, GRAY, rect, 2)

    # Draw Agent
    agent_row, agent_col = agent_pos

    center_x = agent_col * CELL_SIZE + CELL_SIZE // 2
    center_y = agent_row * CELL_SIZE + CELL_SIZE // 2

    pygame.draw.circle(
        screen,
        BLUE,
        (center_x, center_y),
        CELL_SIZE // 4
    )

# -----------------------------
# RUN TRAINED AGENT
# -----------------------------

state = env.reset()

running = True
done = False

while running:

    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_grid(state)

    pygame.display.update()

    if not done:

        row, col = state

        # Best learned action
        action = agent.q_table[row, col].argmax()

        next_state, reward, done = env.step(action)

        state = next_state

        time.sleep(0.4)

    else:
        print("Goal Reached!")
        time.sleep(2)

        state = env.reset()
        done = False

pygame.quit()