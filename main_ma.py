import sys

import pygame
from pettingzoo.utils import parallel_to_aec
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import supersuit as ss

from src.ma_environment import MultiAgentEnvironment


def train(env_fn, steps: int = 100_000, seed: int = 0, **env_kwargs):
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
        learning_rate=0.0003
    )

    model.learn(total_timesteps=steps)

    model.save("ppo_model")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str = None, **env_kwargs):
    env = env_fn(render_mode=render_mode, **env_kwargs)

    env = parallel_to_aec(env)

    model = PPO.load('ppo_model')

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            print(agent, obs)

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()


if __name__ == "__main__":
    env_kwargs = {'width': 10, 'height': 10, 'num_aircraft': 1}
    train(MultiAgentEnvironment, steps=1_000_000, **env_kwargs)

    # run(MultiAgentEnvironment, render_mode='human', **env_kwargs)

    eval(MultiAgentEnvironment, num_games=5, render_mode='human', **env_kwargs)
