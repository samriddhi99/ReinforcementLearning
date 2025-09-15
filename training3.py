import numpy as np
import os
import gymnasium as gym
from RobotNavigation import RobotNavigationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import RewardWrapper
print('new one')
# ------------------- 1. Reward Wrapper -------------------
class SafeRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward <= -1000:
            return -500  # collision penalty
        elif reward >= 900:
            return 1000  # goal reward
        else:
            return reward * 10  # scaled dense reward


# ------------------- 2. Callback -------------------
class LoggingAndSavingCallback(BaseCallback):
    def __init__(self, test_period, test_count, verbose=0):
        super().__init__(verbose)
        self.test_period = test_period
        self.test_count = test_count
        self.episode_reward = 0
        self.training_rewards = []
        self.testing_rewards = []
        self.best_avg_reward = -np.inf

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        self.episode_reward += reward

        if done:
            self.training_rewards.append(self.episode_reward)
            np.save("training_log.npy", self.training_rewards)
            self.episode_reward = 0

        if self.num_timesteps % self.test_period == 0:
            self.model.save("LATEST_MODEL")

            rewards = []
            for _ in range(self.test_count):
                test_env = RobotNavigationEnv()
                test_env = SafeRewardWrapper(test_env)  # wrap test env too
                obs, _ = test_env.reset()
                done, truncated = False, False
                total_reward = 0
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = test_env.step(action)
                    total_reward += reward
                test_env.close()
                rewards.append(total_reward)

            avg_reward = np.mean(rewards)
            self.testing_rewards.append(avg_reward)
            np.save("testing_log.npy", self.testing_rewards)

            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.model.save("BEST_MODEL")

        return True


# ------------------- 3. Training Setup -------------------
os.makedirs("logs2", exist_ok=True)

# Wrap env safely
env = SafeRewardWrapper(RobotNavigationEnv())

# Callback setup
test_period = 20000
test_count = 10
callback = LoggingAndSavingCallback(test_period, test_count)

# Define PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    batch_size=128,
    n_steps=1024,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[128, 128])
)

# Train agent
model.learn(total_timesteps=2_000_000, callback=callback)

# Save final model
model.save("MODEL3")

# Cleanup
env.close()