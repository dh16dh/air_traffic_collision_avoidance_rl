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

        self.position = self.start_position
        self.min_speed = np.array([self.env.min_speed])
        self.max_speed = np.array([self.env.max_speed])
        self.speed = np.array([0.2])
        self.heading = self.get_initial_heading()
        self.done = False

    def update(self, heading_change, speed_change):
        if self.done:
            return 0, True

        self.heading = self.update_heading(heading_change)
        self.speed = self.update_speed(speed_change)
        self.position = self.update_position()

        reward = -0.01

        reward -= float(np.abs(heading_change)) * 0.001

        if np.linalg.norm(self.position - self.end_position) <= self.env.tolerance:
            reward += 10
            self.done = True
            return reward, self.done

        out_of_bounds = (self.position[0] < 0 or self.position[0] > self.env.width or
                         self.position[1] < 0 or self.position[1] > self.env.height)
        if out_of_bounds:
            self.position = np.clip(self.position, [0, 0], [self.env.width, self.env.height])
            reward -= 0.1

        return reward, self.done

    def reset(self):
        self.position = np.array(self.start_position)
        self.heading = np.array(self.get_initial_heading())
        self.done = False

    def get_state(self) -> np.array:
        obs = np.array([self.position[0],
                        self.position[1],
                        self.heading[0],
                        self.speed[0]], dtype=np.float32)
        return obs

    def get_initial_heading(self):
        if self.start_edge == 'top':
            return np.array([180.])
        elif self.start_edge == 'bottom':
            return np.array([0.])
        elif self.start_edge == 'left':
            return np.array([90.])
        elif self.start_edge == 'right':
            return np.array([270.])

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
        new_y = self.position[1] + self.speed[0] * m.cos(theta)

        return new_x, new_y
