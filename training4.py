import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class ModifiedRobotNavigationEnv(gym.Wrapper):
    def __init__(self, env, H):
        super().__init__(env)
        self.H = H
        Delta = H * self.env.delta
        self.action_space = gym.spaces.Box(-Delta * np.ones(2), Delta * np.ones(2), dtype=np.float32)

    def conventional_policy(self, robot_position, goal_intermediate):
        tol = 1e-8
        if np.linalg.norm(robot_position - self.env.goal) <= self.env.goal_radius:
            return -1
        moves = {
        0: np.array([0, 1]),   # up
        1: np.array([0, -1]),  # down
        2: np.array([-1, 0]),  # left
        3: np.array([1, 0]),   # right
    }

        best_action = -1
        min_dist = float('inf')

        for a, move in moves.items():
            candidate = robot_position + move * self.env.delta
            dist = np.linalg.norm(candidate - goal_intermediate)
        #add colision check???
            if dist < min_dist - tol:
                min_dist = dist
                best_action = a
        return best_action

    def step(self, action):
        goal_intermediate = self.env.robot_position + action
        grid_x = int(goal_intermediate[0] / self.env.delta) + 1
        grid_y = int(goal_intermediate[1] / self.env.delta) + 1
        grid_x = self.env.delta * (0.5 + (grid_x - 1))
        grid_y = self.env.delta * (0.5 + (grid_y - 1))
        goal_intermediate = np.array([grid_x, grid_y])

        reward_miniepisode = 0
        for h in range(self.H):
            reward = -np.sqrt(np.sum((self.env.robot_position - self.env.goal) ** 2))
            a = self.conventional_policy(self.env.robot_position, goal_intermediate)

            if a != -1:
                self.env.robot_position = self.env.robot_position + self.env.action_dict[a] * self.env.delta

            self.env.trail.append(self.env.robot_position)

            terminated = self.env.check_collision()
            if terminated:
                reward = -10000

            if not terminated and np.sum((self.env.goal - self.env.robot_position) ** 2) <= self.env.goal_radius ** 2:
                terminated = True
                reward = 1000

            reward_miniepisode += reward

            self.env.t += 1
            truncated = self.env.t > self.env.Horizon

            if self.env.render_mode == "human":
                self.env.render()

            if terminated or truncated:
                break

        self.env.observation = np.concatenate((self.env.get_lidar_reading(), self.env.robot_position))
        return self.env.observation, reward_miniepisode, terminated, truncated, {}


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
                test_env = ModifiedRobotNavigationEnv(RobotNavigationEnv(), H=40)
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


env = RobotNavigationEnv()
H = 40
env = ModifiedRobotNavigationEnv(env, H)

test_period = 20000
test_count = 10
callback = LoggingAndSavingCallback(test_period, test_count)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=2e-4,
    batch_size=128,
    n_steps=2048,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    policy_kwargs=dict(net_arch=[128, 128])
)

model.learn(total_timesteps=200_000, callback=callback)  #had to decrease due to time constraints, otherwise wouldnt get model to submit on time

env.close()

model.save("MODEL4")