import numpy as np
import gymnasium as gym
import pygame
# Write just ONE line of code below this comment to import DQN from stable baseline
from stable_baselines3 import DQN

def visualize_model_performance(model):
    env = gym.make('MountainCar-v0', render_mode='human')
    
    x, _ = env.reset()
    total_reward = 0
    terminated, truncated = False, False
    while not(terminated) and not(truncated):
        action, _ = model.predict(x)
        x, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    
    print('Total reward = {}'.format(total_reward))
    env.close()
    # pygame.display.quit() # Use this line when the display screen is not going away
        
class Custom_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        position = observation[0]
        speed = abs(observation[1])
        #print("[Wrapper] position:", position, "terminated:", terminated) 
        shaped_reward = reward + max(0, observation[1])  # 
        if terminated and position >= 0.5:
            print("FLAG REACHED - GIVING BONUS!") 
            shaped_reward += 100000  # Try a huge bonus!
        return observation, shaped_reward, terminated, truncated, info

env = gym.make('MountainCar-v0')
env = Custom_Wrapper(env)
model = DQN("MlpPolicy", env, learning_rate=0.001, buffer_size=50000, learning_starts=1000,
            batch_size=64, tau=0.8, gamma=0.98, train_freq=4, target_update_interval=500,
            exploration_initial_eps=1.0, exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[64, 64]), verbose=1)
model.learn(total_timesteps=120000)
model.save("MODEL2")

visualize_model_performance(DQN.load("MODEL2.zip"))