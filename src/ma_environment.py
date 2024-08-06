import functools
from copy import copy

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID
from pettingzoo.test import parallel_api_test

from src.ma_airplane import Aircraft
from src.visualizer import PygameVisualizer


class MultiAgentEnvironment(ParallelEnv):
    metadata = {'render_modes': ['human'], 'name': 'multi_agent_aircraft_collision_avoidance'}

    def __init__(self, width: int, height: int, num_aircraft: int,
                 max_hdg_chg: float = 5.0, max_spd_chg: float = 0.0005, render_mode: str = None):
        # Environment specifications
        self.env_width = width
        self.env_height = height

        self.timestep = None

        # Action space
        self.heading_multiplier = max_hdg_chg
        self.speed_multiplier = max_spd_chg

        self.possible_agents = [Aircraft(i, self) for i in range(num_aircraft)]

        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.visualizer = PygameVisualizer(self)

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        for agent in self.agents:
            agent.reset()

        observations = {
            agent: (
                agent.position[0],
                agent.position[1],
                agent.velocity[0],
                agent.velocity[1],
                agent.heading,
                agent.speed,
                agent.dist_to_goal,
                agent.dist_to_ideal_track
            )
            for agent in self.agents
        }

        infos = {
            agent: {}
            for agent in self.agents
        }

        return observations, infos

    def step(self, actions):
        self.render()

        for agent in self.agents:
            hdg_chg, spd_chg = actions[agent]
            hdg_chg = self.heading_multiplier * hdg_chg
            spd_chg = self.speed_multiplier * spd_chg
            agent.step(hdg_chg, spd_chg)

        terminations = {agent: agent.terminated for agent in self.agents}
        rewards = {agent: agent.reward for agent in self.agents}

        truncations = {agent: False for agent in self.agents}
        if self.timestep > 10000:
            truncations = {agent: True for agent in self.agents}
        self.timestep += 1

        observations = {
            agent: (
                agent.position[0],
                agent.position[1],
                agent.velocity[0],
                agent.velocity[1],
                agent.heading,
                agent.speed,
                agent.dist_to_goal,
                agent.dist_to_ideal_track
            )
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        self.agents = [agent for agent in self.agents if not agent.terminated]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == 'human':
            self.visualizer.check_for_end()
            self.visualizer.render()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        min_pos_x, max_pos_x = 0, self.env_width
        min_pos_y, max_pos_y = 0, self.env_height
        min_dist_to_goal, max_dist_to_goal = 0, np.hypot(self.env_width, self.env_height)
        min_dist_ideal, max_dist_ideal = 0, np.hypot(self.env_width, self.env_height)
        min_hdg, max_hdg = 0, 360
        min_spd, max_spd = 0.005, 0.01
        min_v_x = min_v_y = min_spd
        max_v_x = max_v_y = max_spd

        min_obs_array = [min_pos_x, min_pos_y, min_v_x, min_v_y, min_hdg, min_spd, min_dist_to_goal, min_dist_ideal]
        max_obs_array = [max_pos_x, max_pos_y, max_v_x, max_v_y, max_hdg, max_spd, max_dist_to_goal, max_dist_ideal]

        assert len(min_obs_array) == len(max_obs_array)

        return spaces.Box(low=np.array(min_obs_array),
                          high=np.array(max_obs_array),
                          shape=(len(min_obs_array),),
                          dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        return spaces.Box(low=np.array([-1, -1]),
                          high=np.array([1, 1]),
                          shape=(2,),
                          dtype=np.float32)

    def close(self):
        if self.render_mode == 'human':
            self.visualizer.close()


if __name__ == "__main__":
    env = MultiAgentEnvironment(10, 10, 3)
    parallel_api_test(env, num_cycles=1_000_000)
