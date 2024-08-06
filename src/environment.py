from __future__ import annotations

import random

import numpy as np
import gymnasium as gym
import pygame

from gymnasium import spaces
from src.airplane import Aircraft
from src.edge import Edge


class Environment(gym.Env):
    def __init__(self, width: int, height: int, max_hdg_chg: float, max_spd_chg: float, min_speed: float,
                 max_speed: float, observation_space: spaces.Box, action_space: spaces.Box,
                 num_aircraft: int,
                 tolerance=0.5):
        self.width = width
        self.height = height
        self.tolerance = tolerance
        self.num_aircraft = num_aircraft
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.aircraft_list = self.create_aircraft()

        # Define observation space
        self.observation_space = observation_space

        # Define action space
        self.action_space = action_space
        self.heading_multiplier = max_hdg_chg
        self.speed_multiplier = max_spd_chg

        self.action_num = int(self.action_space.shape[0] / self.num_aircraft)

    def step(self, action):
        rewards = []
        terminated = []
        states = []
        for i, aircraft in enumerate(self.aircraft_list):
            heading_change = action[i * self.action_num] * self.heading_multiplier
            speed_change = action[i * self.action_num + 1] * self.speed_multiplier
            reward, done = aircraft.update(heading_change, speed_change)
            states.extend(aircraft.get_state())
            rewards.append(reward)
            terminated.append(done)

        return np.array(states), np.sum(rewards), all(terminated), False, {}

    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        super().reset(seed=seed)

        states = []
        for aircraft in self.aircraft_list:
            aircraft.reset()
            states.extend(aircraft.get_state())
        return np.array(states), {}

    def create_aircraft(self):
        aircraft = []
        for n in range(self.num_aircraft):
            while True:
                start_edge = self._generate_random_edge()
                end_edge = self._generate_random_edge()
                if not start_edge.kind == end_edge.kind:
                    break
            aircraft.append(Aircraft(start_edge, end_edge, self))
        return aircraft

    def _generate_random_edge(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        edge_obj = Edge(edge, self.width, self.height)
        return edge_obj
