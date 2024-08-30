import functools
import heapq
from copy import copy

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID
from pettingzoo.test import parallel_api_test

from src.ma_airplane import Aircraft
from src.visualizer import PygameVisualizer
from utils.normalizers import normalize_range_min1_plus1, normalize_range_0_1
from utils.units import nmi_to_km, kmh_to_kms


class MultiAgentEnvironment(ParallelEnv):
    metadata = {'render_modes': ['human'], 'name': 'multi_agent_aircraft_collision_avoidance'}

    MAX_NEARBY_AGENTS = 3

    def __init__(self, width: int, height: int, num_aircraft: int, render_mode: str = None, **kwargs):
        # Environment specifications
        self.env_width = width
        self.env_height = height

        self.timestep = None
        if 'seed' in kwargs.keys():
            self.seed = kwargs['seed']

        # Observation space
        self.min_pos_x, self.max_pos_x = 0, self.env_width
        self.min_pos_y, self.max_pos_y = 0, self.env_height
        self.min_dist_to_goal, self.max_dist_to_goal = 0, np.hypot(self.env_width, self.env_height)
        self.min_dist_ideal, self.max_dist_ideal = 0, np.hypot(self.env_width, self.env_height)
        self.min_hdg, self.max_hdg = -180, 180
        self.min_spd, self.max_spd = kmh_to_kms(700), kmh_to_kms(900)
        self.min_v_x = self.min_v_y = -self.max_spd
        self.max_v_x = self.max_v_y = self.max_spd

        self.num_base_observations = 7
        self.num_states_per_nearby = 4
        self.num_observations = self.num_base_observations + self.num_states_per_nearby * self.MAX_NEARBY_AGENTS
        # Action space
        # Heading changes based on standard turn rate being 3deg/s.
        # Speed changes to be determined.
        self.heading_changes = np.arange(-4.5, 4.6, 0.5)
        self.speed_changes = np.arange(kmh_to_kms(-50), kmh_to_kms(51), kmh_to_kms(50))

        self.possible_agents = [Aircraft(i, self) for i in range(num_aircraft)]

        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.visualizer = PygameVisualizer(self)

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        for idx, agent in enumerate(self.agents):
            if isinstance(seed, int):
                seed = seed + idx * 999
            else:
                seed = seed
            agent.reset(seed)

        observations = {
            agent: self.get_observations(agent)
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
            hdg_chg, spd_chg = self.action_map(actions[agent])
            agent.step(hdg_chg, spd_chg)

        for idx_1, agent_1 in enumerate(self.possible_agents):
            if agent_1 not in self.agents:
                continue
            for idx_2, agent_2 in enumerate(self.possible_agents):
                if agent_2 not in self.agents or agent_1 == agent_2:
                    continue
                distance = self.calculate_distances_between_agents(agent_1, agent_2)
                if distance < nmi_to_km(10):
                    heapq.heappush(agent_1.nearby_aircraft, (distance, agent_2))
                    if len(agent_1.nearby_aircraft) > 3:
                        agent_1.nearby_aircraft = agent_1.nearby_aircraft[:3]

        terminations = {agent: agent.terminated for agent in self.agents}
        rewards = {agent: agent.reward for agent in self.agents}

        truncations = {agent: False for agent in self.agents}
        if self.timestep > 5_000:
            truncations = {agent: True for agent in self.agents}
        self.timestep += 1

        observations = {
            agent: self.get_observations(agent)
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
        return spaces.Box(low=-1,
                          high=1,
                          shape=(self.num_observations,),
                          dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        return spaces.MultiDiscrete([len(self.heading_changes), len(self.speed_changes)])

    def close(self):
        if self.render_mode == 'human':
            self.visualizer.close()

    def get_observations(self, agent):
        base_observation = (
            normalize_range_0_1(agent.dist_to_goal, self.min_dist_to_goal, self.max_dist_to_goal),
            normalize_range_0_1(agent.dist_to_ideal_track, self.min_dist_ideal, self.max_dist_ideal),
            normalize_range_0_1(agent.speed, self.min_spd, self.max_spd),
            normalize_range_min1_plus1(agent.position[0], self.min_pos_x, self.max_pos_x),
            normalize_range_min1_plus1(agent.position[1], self.min_pos_y, self.max_pos_y),
            normalize_range_min1_plus1(agent.rel_heading, self.min_hdg, self.max_hdg),
            normalize_range_min1_plus1(agent.vel_towards_goal, -self.max_spd, self.max_spd)
        )
        nearby_observations = ()
        for distance, other in agent.nearby_aircraft:
            rel_v_long, rel_v_lat = agent.get_relative_velocity_vectors(other)
            rel_angle = agent.get_relative_position_angle(other)
            nearby_observation = (
                normalize_range_min1_plus1(rel_v_lat, 2 * self.min_v_x, 2 * self.max_v_x),
                normalize_range_min1_plus1(rel_v_long, 2 * self.min_v_y, 2 * self.max_v_y),
                normalize_range_0_1(distance, 0, nmi_to_km(10)),
                normalize_range_min1_plus1(rel_angle, self.min_hdg, self.max_hdg)
            )
            nearby_observations += nearby_observation
        base_observation += nearby_observations
        while len(base_observation) < self.num_observations:
            base_observation += tuple([0])
        return base_observation

    def action_map(self, actions) -> tuple[float, float]:
        heading_index, speed_index = actions
        return self.heading_changes[heading_index], self.speed_changes[speed_index]

    @staticmethod
    def calculate_distances_between_agents(agent_1: AgentID, agent_2: AgentID) -> float:
        try:
            distance = np.linalg.norm(agent_1.position - agent_2.position)
        except ZeroDivisionError:
            distance = 1e-99
        return distance


if __name__ == "__main__":
    env = MultiAgentEnvironment(10, 10, 3)
    parallel_api_test(env, num_cycles=1_000_000)
