from __future__ import annotations

import math as m
import random

import numpy as np

from src.edge import Edge
from utils.normalizers import normalize_range_0_1
from utils.units import nmi_to_km, kmh_to_kms


class Aircraft:
    PAZ = nmi_to_km(5)
    NMAC = nmi_to_km(0.1)

    def __init__(self, name, env):
        self.name = f"Aircraft_{str(name)}"
        self.env = env

        self.start_position = None
        self.end_position = None
        self.tolerance = 0.5

        # Hidden States
        self.position = None
        self.velocity = None
        self.true_heading = None
        self.timestep = None

        # States
        self.speed = None
        self.vel_towards_goal = None
        self.rel_heading = None
        self.dist_to_goal = None
        self.dist_to_ideal_track = None

        self.nearby_aircraft = []

        self.terminated = False
        self.reward = None

    def __str__(self):
        return f"{self.name}"

    def __lt__(self, other):
        return None

    def reset(self, seed):
        self.start_position, self.end_position = self.get_start_end_points(seed)

        self.position = self.start_position
        self.speed = kmh_to_kms(800)
        self.true_heading = self.get_ideal_heading()
        self.rel_heading = 0
        self.vel_towards_goal = self.get_relative_velocity_to_goal()
        self.velocity = self.update_velocity()
        self.dist_to_goal = self.get_distance_to_goal()
        self.dist_to_ideal_track = self.get_distance_to_ideal_track()

        self.nearby_aircraft = []

        self.terminated = False
        self.reward = 0
        self.timestep = 0

    def step(self, heading_change, speed_change):
        self.true_heading = self.update_heading(heading_change)
        self.speed = self.update_speed(speed_change)
        self.position = self.update_position()
        self.velocity = self.update_velocity()
        self.rel_heading = self.true_heading - self.get_ideal_heading()
        self.vel_towards_goal = self.get_relative_velocity_to_goal()
        self.dist_to_goal = self.get_distance_to_goal()
        self.dist_to_ideal_track = self.get_distance_to_ideal_track()

        dist_to_ideal_norm = normalize_range_0_1(self.dist_to_ideal_track, nmi_to_km(2),
                                                 self.env.max_dist_ideal) * 10
        hdg_chg_norm = normalize_range_0_1(abs(heading_change), 0, max(self.env.heading_changes)) * 0.5

        self.reward = 0
        self.reward -= 0.1
        self.reward -= dist_to_ideal_norm
        self.reward -= hdg_chg_norm

        while self.nearby_aircraft:
            distance, other_agent = self.nearby_aircraft.pop()
            if distance < self.NMAC:
                self.terminated = True
            else:
                self.reward += self.loss_of_separation_reward(distance)

        if self.dist_to_goal < self.tolerance:
            self.reward += 100
            self.terminated = True

        out_of_bounds = (self.position[0] < 0 or self.position[0] > self.env.env_width or
                         self.position[1] < 0 or self.position[1] > self.env.env_height)
        if out_of_bounds and self.timestep > 0:
            self.position = np.clip(self.position, [0., 0.], [self.env.env_width, self.env.env_height])
            self.reward -= 15
            # self.terminated = True

        self.timestep += 1

    def get_start_end_points(self, seed):
        try:
            seed2 = seed + 1
        except TypeError as e:
            seed2 = seed
        while True:
            start_edge = self._generate_random_edge(seed)
            end_edge = self._generate_random_edge(seed2)
            if isinstance(seed2, int):
                seed2 += 1
            if not start_edge.kind == end_edge.kind:
                break
        start_position = np.array([start_edge.x, start_edge.y])
        end_position = np.array([end_edge.x, end_edge.y])
        return start_position, end_position

    def _generate_random_edge(self, seed):
        random.seed(seed)
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        edge_obj = Edge(edge, self.env.env_width, self.env.env_height, seed)
        return edge_obj

    def get_ideal_heading(self):
        d_x = self.end_position[0] - self.position[0]
        d_y = self.end_position[1] - self.position[1]
        heading = (90 - np.rad2deg(np.arctan2(d_y, d_x))) % 360
        return heading

    def get_distance_to_goal(self):
        return np.abs(np.linalg.norm(self.end_position - self.position))

    def get_distance_to_ideal_track(self):
        direction_vector = self.end_position - self.start_position
        start_to_current = self.position - self.start_position

        project_scalar = np.dot(start_to_current, direction_vector) / np.dot(direction_vector, direction_vector)
        project_vector = project_scalar * direction_vector

        closest_point_on_track = self.start_position + project_vector

        distance_to_track = np.linalg.norm(self.position - closest_point_on_track)

        return distance_to_track

    def update_heading(self, hdg_chg):
        new_heading = (self.true_heading + hdg_chg) % 360
        return new_heading

    def update_speed(self, spd_chg):
        new_speed = (self.speed + spd_chg)
        new_speed = np.clip(new_speed, kmh_to_kms(700), kmh_to_kms(900))
        return new_speed

    def update_position(self):
        theta = np.deg2rad(self.true_heading)

        new_x = self.position[0] + self.speed * m.sin(theta)
        new_y = self.position[1] + self.speed * m.cos(theta)

        return np.array([new_x, new_y])

    def update_velocity(self):
        theta = np.deg2rad(self.true_heading)
        v_x, v_y = self.speed * m.sin(theta), self.speed * m.cos(theta)
        return np.array([v_x, v_y])

    @staticmethod
    def loss_of_separation_reward(distance):
        reward = -nmi_to_km(2.5) + distance * 0.25
        # if self.PAZ >= distance:
        #     reward = -nmi_to_km(5) + distance
        #     # try:
        #     #     reward = - nmi_to_km(2.5) / (0.5 + distance)
        #     # except ZeroDivisionError:
        #     #     reward = - nmi_to_km(5)
        # else:
        #     reward = 0
        return reward

    def get_relative_velocity_vectors(self, other: Aircraft):
        relative_v_vector = other.velocity - self.velocity
        v_long_vec = np.dot(relative_v_vector, self.velocity) / np.dot(self.velocity, self.velocity) * self.velocity
        v_lat_vec = relative_v_vector - v_long_vec

        u_long = self.velocity / np.linalg.norm(self.velocity)
        u_lat = np.array([-u_long[1], u_long[0]])
        T = np.column_stack((u_long, u_lat)).T
        # TODO: Fix ordering
        v_long = np.matmul(T, v_long_vec)[0]
        v_lat = -np.matmul(T, v_lat_vec)[1]

        return v_long, v_lat

    def get_relative_position_angle(self, other: Aircraft):
        heading_other = np.arctan2(other.position[1] - self.position[1], other.position[0] - self.position[0])
        heading_other = 90 - np.rad2deg(heading_other)
        relative_heading = heading_other - self.true_heading
        relative_heading = (relative_heading + 180) % 360 - 180
        return relative_heading

    def get_relative_velocity(self, other: Aircraft):
        return other.velocity - self.velocity

    def get_relative_position(self, other: Aircraft):
        return other.position - self.position

    def get_relative_velocity_to_goal(self):
        hdg = np.deg2rad(self.rel_heading)
        return self.speed * m.cos(hdg)
