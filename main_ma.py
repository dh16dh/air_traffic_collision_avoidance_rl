import random
import sys

import pygame
from pettingzoo.utils import parallel_to_aec
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import supersuit as ss

from src.ma_environment import MultiAgentEnvironment


def train(env_fn, steps: int = 100_000, seed: int = None, **env_kwargs):
    env = env_fn(**env_kwargs)

    env = ss.black_death_v3(env)
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        gamma=0.999,
        tensorboard_log="logs"
    )

    model.learn(total_timesteps=steps)

    model.save("ppo_model")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str = None, **env_kwargs):
    env = env_fn(render_mode=render_mode, **env_kwargs)

    env = parallel_to_aec(env)

    model = PPO.load('ppo_model')

    for i in range(num_games):
        rewards = {str(agent): 0 for agent in env.possible_agents}
        env.reset(seed=i)

        terminations = {agent: False for agent in env.possible_agents}
        truncations = {agent: False for agent in env.possible_agents}

        times = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            terminations[agent] = termination
            truncations[agent] = truncation

            for a in env.agents:
                rewards[str(a)] += env.rewards[a]
            if not terminations[agent] and not truncations[agent]:
                act = model.predict(obs, deterministic=True)[0]
                env.step(act)
            else:
                env.step(None)

            if all(terminations.values()) or all(truncations.values()):
                times = agent.timestep
                break
        print(f"Game {i} / {num_games}: Rewards: {rewards}, Timesteps: {times}")

    env.close()


if __name__ == "__main__":
    env_kwargs = {'width': 500, 'height': 500, 'num_aircraft': 15}
    train(MultiAgentEnvironment, steps=7_000_000, **env_kwargs)

    eval(MultiAgentEnvironment, num_games=10, render_mode="human", **env_kwargs)
