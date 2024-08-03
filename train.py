import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from job_opportunities_env import JobOpportunitiesEnv

# Initialize the environment
env = JobOpportunitiesEnv()
env = DummyVecEnv([lambda: env])  # Wrap the environment

# Define and train the model
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)

# Save the model
model.save("dqn_policy_network")
print("Model saved successfully.")

# Load the model (for testing purposes)
loaded_model = DQN.load("dqn_policy_network")
print("Model loaded successfully.")

# Evaluate the model (optional)
obs = env.reset()
for i in range(1000):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
