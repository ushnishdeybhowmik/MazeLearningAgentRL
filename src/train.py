from maze import Maze
from agent import QLearningAgent

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

print(agent.q_table)