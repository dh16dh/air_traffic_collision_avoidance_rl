import random
import sys
from copy import copy

import pygame
from pettingzoo.utils import parallel_to_aec
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.ppo import MlpPolicy

import supersuit as ss

from src.ma_environment import MultiAgentEnvironment


def train(env_fn, steps_single: int = 100_000, steps_multi: int = 100_000, seed: int = None, **env_kwargs):
    width = env_kwargs.get("width")
    height = env_kwargs.get("height")
    num_agents_single = env_kwargs.get("num_aircraft_single")
    num_agents_multi = env_kwargs.get("num_aircraft_multi")

    env_kwargs_single = {'width': width,
                         'height': height,
                         'num_aircraft': num_agents_single}
    env_kwargs_multi = {'width': width,
                        'height': height,
                        'num_aircraft': num_agents_multi}

    env = env_fn(**env_kwargs_single)
    env = ss.black_death_v3(env)
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    eval_env = copy(env)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=50, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=10240, callback_after_eval=stop_train_callback, verbose=1)

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        n_epochs=20,
        gamma=0.999,
        tensorboard_log="logs"
    )

    model.learn(total_timesteps=steps_single, callback=eval_callback)
    model.save('ppo_model_single')
    env = env_fn(**env_kwargs_multi)
    env = ss.black_death_v3(env)
    env.reset(seed=seed)

    print(f"Starting multi-agent training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    model.learn(total_timesteps=steps_multi)

    model.save("ppo_model")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str = None, **env_kwargs):
    env_kwargs = {'width': env_kwargs.get('width'),
                  'height': env_kwargs.get('height'),
                  'num_aircraft': env_kwargs.get('num_aircraft_multi')}
    env = env_fn(render_mode=render_mode, **env_kwargs)

    env = parallel_to_aec(env)

    model = PPO.load('ppo_model')

    for i in range(num_games):
        rewards = {str(agent): 0 for agent in env.possible_agents}
        env.reset(seed=i)

        terminations = {agent: False for agent in env.possible_agents}
        truncations = {agent: False for agent in env.possible_agents}
        PAZ_incursions = {str(agent): agent.num_PAZ_incursion for agent in env.possible_agents}
        NMAC_incursions = {str(agent): agent.num_NMAC_incursion for agent in env.possible_agents}
        path_efficiency = {str(agent): 0 for agent in env.possible_agents}
        times = {str(agent): 0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            terminations[agent] = termination
            truncations[agent] = truncation
            PAZ_incursions[str(agent)] = agent.num_PAZ_incursion
            NMAC_incursions[str(agent)] = agent.num_NMAC_incursion

            for a in env.agents:
                rewards[str(a)] += env.rewards[a]
            if not terminations[agent] and not truncations[agent]:
                act = model.predict(obs, deterministic=True)[0]
                env.step(act)
            else:
                path_efficiency[str(agent)] = agent.calculate_path_efficiency()
                times[str(agent)] = agent.timestep
                env.step(None)

            if all(terminations.values()) or all(truncations.values()):
                path_efficiency[str(agent)] = agent.calculate_path_efficiency()
                break

        reward_per_timestep = {str(agent): rewards[str(agent)] / times[str(agent)] for agent in env.possible_agents}

        print(f"Game {i} / {num_games}: Rewards: {rewards}")
        print(f"Game {i} / {num_games}: PAZ Incursions: {PAZ_incursions}")
        print(f"Game {i} / {num_games}: NMAC Incursions: {NMAC_incursions}")
        print(f"Game {i} / {num_games}: Path Efficiency: {path_efficiency}")
        print(f"Game {i} / {num_games}: Time Taken: {times}")
        print(f"Game {i} / {num_games}: Rewards per Timestep: {reward_per_timestep}")
        print()

    env.close()


if __name__ == "__main__":
    env_kwargs = {'width': 400, 'height': 400, 'num_aircraft_single': 1, 'num_aircraft_multi': 10}
    train(MultiAgentEnvironment,
          steps_single=5000,
          steps_multi=0,
          **env_kwargs)

    eval(MultiAgentEnvironment, num_games=10, render_mode="human", **env_kwargs)
