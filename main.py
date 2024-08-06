import sys

import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from hyperparam_tuner import HyperparamTuner
from src.environment import Environment
from src.visualizer import PygameVisualizer

width, height = 10, 10
max_hdg_chg = 15
max_spd_chg = 0.01
min_spd, max_spd = 0.1, 0.3

max_dist = np.hypot(height, width)

state_space_minimum = np.array([0, 0])
state_space_maximum = np.array([max_dist, max_dist])
action_space_minimum = np.array([-1, -1])
action_space_maximum = np.array([1, 1])

num_aircraft = 5

observation_space = spaces.Box(low=np.tile(state_space_minimum, num_aircraft),
                               high=np.tile(state_space_maximum, num_aircraft),
                               shape=(num_aircraft * len(state_space_minimum), ),
                               dtype=np.float32)

action_space = spaces.Box(low=np.tile(action_space_minimum, num_aircraft),
                          high=np.tile(action_space_maximum, num_aircraft),
                          shape=(num_aircraft * len(action_space_minimum), ),
                          dtype=np.float32)

env = Environment(width, height, max_hdg_chg, max_spd_chg, min_spd, max_spd, num_aircraft=num_aircraft,
                  observation_space=observation_space, action_space=action_space)

check_env(env, warn=True)

# tuner = HyperparamTuner(env)
# best_params = tuner.tune()

agent = PPO('MlpPolicy', env, verbose=1)
agent.learn(total_timesteps=1000000)

agent.save('ppo_model')

del agent

model = PPO.load('ppo_model')
visualizer = PygameVisualizer(env)

obs, info = env.reset()
done = False

# Run the trained model
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    action, _states = model.predict(obs)
    obs, rewards, terminated, done, info = env.step(action)

    print(f"Dist to Goal: {obs[0]}, Dist to Track: {obs[1]}, HdgChg: {action[0]*max_hdg_chg}, SpdChg: {action[1]*max_spd_chg}, Reward: {rewards}")

    visualizer.render()
    visualizer.clock.tick(7)

    if terminated:
        obs, info = env.reset()
        done = True

visualizer.close()
