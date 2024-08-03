import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from job_opportunities_env import JobOpportunitiesEnv

# Create the environment
env = DummyVecEnv([lambda: JobOpportunitiesEnv()])

# Define the model
model = DQN('MlpPolicy', env, learning_rate=1e-3, buffer_size=50000, learning_starts=10, target_update_interval=100, verbose=1)

# Train the model
model.learn(total_timesteps=50000)

# Save the model
model.save("dqn_job_opportunities")

# Load the model
model = DQN.load("dqn_job_opportunities", env=env)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
