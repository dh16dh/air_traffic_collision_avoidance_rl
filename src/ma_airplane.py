import random
import math as m

import numpy as np

from src.edge import Edge


class Aircraft:
    def __init__(self, name, env):
        self.name = f"Aircraft_{str(name)}"
        self.env = env

        self.start_position = None
        self.end_position = None
        self.tolerance = 0.5

        # States
        self.position = None
        self.speed = None
        self.velocity = None
        self.heading = None
        self.dist_to_goal = None
        self.dist_to_ideal_track = None

        self.terminated = False
        self.reward = None

    def __str__(self):
        return f"{self.name}"

    def reset(self):
        self.start_position, self.end_position = self.get_start_end_points()

        self.position = self.start_position
        self.speed = 0.007
        self.heading = self.get_initial_heading()
        self.velocity = self.update_velocity()
        self.dist_to_goal = self.get_distance_to_goal()
        self.dist_to_ideal_track = self.get_distance_to_ideal_track()

        self.terminated = False
        self.reward = 0

    def step(self, heading_change, speed_change):
        self.heading = self.update_heading(heading_change)
        self.speed = self.update_speed(speed_change)
        self.position = self.update_position()
        self.velocity = self.update_velocity()
        self.dist_to_goal = self.get_distance_to_goal()
        self.dist_to_ideal_track = self.get_distance_to_ideal_track()

        self.reward -= 0.1

        self.reward -= np.abs(heading_change)

        if self.dist_to_goal < self.tolerance:
            self.reward += 50
            self.terminated = True

        out_of_bounds = (self.position[0] < 0 or self.position[0] > self.env.env_width or
                         self.position[1] < 0 or self.position[1] > self.env.env_height)
        if out_of_bounds:
            self.position = np.clip(self.position, [0., 0.], [self.env.env_width, self.env.env_height])
            self.reward -= 0.5

    def get_start_end_points(self):
        while True:
            start_edge = self._generate_random_edge()
            end_edge = self._generate_random_edge()
            if not start_edge.kind == end_edge.kind:
                break
        start_position = np.array([start_edge.x, start_edge.y])
        end_position = np.array([end_edge.x, end_edge.y])
        return start_position, end_position

    def _generate_random_edge(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        edge_obj = Edge(edge, self.env.env_width, self.env.env_height)
        return edge_obj

    def get_initial_heading(self):
        d_x = self.end_position[0] - self.start_position[0]
        d_y = self.end_position[1] - self.start_position[1]
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
        new_heading = (self.heading + hdg_chg) % 360
        return new_heading

    def update_speed(self, spd_chg):
        new_speed = (self.speed + spd_chg)
        new_speed = np.clip(new_speed, 0.005, 0.01)
        return new_speed

    def update_position(self):
        theta = np.deg2rad(self.heading)

        new_x = self.position[0] + self.speed * m.sin(theta)
        new_y = self.position[1] + self.speed * m.cos(theta)

        return np.array([new_x, new_y])

    def update_velocity(self):
        theta = np.deg2rad(self.heading)
        v_x, v_y = self.speed * m.sin(theta), self.speed * m.cos(theta)
        return np.array([v_x, v_y])
