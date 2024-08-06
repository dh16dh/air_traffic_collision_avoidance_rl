from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import math as m

if TYPE_CHECKING:
    from src.environment import Environment
    from src.edge import Edge


class Aircraft:

    def __init__(self, start_position: Edge, end_position: Edge, env: Environment):
        self.start_position = np.array([start_position.x, start_position.y])
        self.start_edge = start_position.kind
        self.end_position = np.array([end_position.x, end_position.y])
        self.env = env

        # Limits
        self.min_speed = np.array([self.env.min_speed])
        self.max_speed = np.array([self.env.max_speed])

        # States
        self.position = self.start_position
        self.speed = np.array([0.2])
        self.heading = self.get_initial_heading()
        self.dist_to_goal = self.get_distance_to_goal()
        self.dist_to_ideal_track = self.get_distance_to_ideal_track()
        self.prev_dist_to_goal = self.dist_to_goal
        self.time_step = 0
        self.done = False

    def update(self, heading_change, speed_change):
        if self.done:
            return 0, True

        self.heading = self.update_heading(heading_change)
        self.speed = self.update_speed(speed_change)
        self.position = self.update_position()
        self.dist_to_goal = self.get_distance_to_goal()
        self.dist_to_ideal_track = self.get_distance_to_ideal_track()
        self.time_step += 1

        reward = -0.01

        reward -= self.dist_to_goal * 0.01
        reward -= self.dist_to_ideal_track * 0.1

        if self.dist_to_goal < self.prev_dist_to_goal:
            reward += 0.1 * self.dist_to_goal

        self.prev_dist_to_goal = self.dist_to_goal

        if np.linalg.norm(self.position - self.end_position) <= self.env.tolerance:
            reward += 50
            self.done = True
            return reward, self.done

        out_of_bounds = (self.position[0] < 0 or self.position[0] > self.env.width or
                         self.position[1] < 0 or self.position[1] > self.env.height)
        if out_of_bounds:
            self.position = np.clip(self.position, [0, 0], [self.env.width, self.env.height])
            reward -= 0.5

        return reward, self.done

    def reset(self):
        self.position = np.array(self.start_position)
        self.heading = np.array(self.get_initial_heading())
        self.speed = np.array([0.2])
        self.dist_to_goal = self.get_distance_to_goal()
        self.dist_to_ideal_track = self.get_distance_to_ideal_track()
        self.time_step = 0
        self.done = False

    def get_state(self) -> np.array:
        obs = np.array([self.dist_to_goal,
                        self.dist_to_ideal_track,
                        ], dtype=np.float32)
        return obs

    def get_initial_heading(self):
        d_x = self.end_position[0] - self.start_position[0]
        d_y = -self.end_position[1] + self.start_position[1]
        heading = np.rad2deg(np.arctan2(d_y, d_x)) % 360
        return np.array([heading])

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
        new_speed = np.clip(new_speed, self.min_speed, self.max_speed)
        return new_speed

    def update_position(self):
        theta = np.deg2rad(self.heading)

        new_x = self.position[0] + self.speed[0] * m.sin(theta)
        new_y = self.position[1] + self.speed[0] * -m.cos(theta)

        return new_x, new_y
