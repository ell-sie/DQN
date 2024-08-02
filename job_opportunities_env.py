import gym
from gym import spaces
import numpy as np

class JobOpportunitiesEnv(gym.Env):
    def __init__(self):
        super(JobOpportunitiesEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=5, shape=(5, 5), dtype=int)

        self.reset()

    def reset(self):
        self.state = np.zeros((5, 5))
        self.state[0, 0] = 1  # Agent starts at (0, 0)
        self.state[4, 4] = 2  # Goal position (e.g., job opportunity center)
        return self.state

    def step(self, action):
        agent_position = np.argwhere(self.state == 1)[0]
        self.state[agent_position[0], agent_position[1]] = 0

        if action == 0:  # Up
            agent_position[0] = max(agent_position[0] - 1, 0)
        elif action == 1:  # Down
            agent_position[0] = min(agent_position[0] + 1, 4)
        elif action == 2:  # Left
            agent_position[1] = max(agent_position[1] - 1, 0)
        elif action == 3:  # Right
            agent_position[1] = min(agent_position[1] + 1, 4)

        self.state[agent_position[0], agent_position[1]] = 1

        done = (agent_position[0] == 4 and agent_position[1] == 4)
        reward = 1 if done else -0.1  # Reward for reaching the goal, penalty otherwise

        return self.state, reward, done, {}

    def render(self, mode='human'):
        for row in self.state:
            print(' '.join(map(str, row)))
        print("\n")

if __name__ == '__main__':
    env = JobOpportunitiesEnv()
    state = env.reset()
    env.render()

    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
