import sys
import numpy as np
import pygame


class PygameVisualizer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.width = env.width * 100
        self.height = env.height * 100
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Aircraft Navigation")
        self.clock = pygame.time.Clock()

    def draw_aircraft(self):
        for aircraft in self.env.aircraft_list:
            pos = aircraft.position
            end_pos = aircraft.end_position
            pygame.draw.circle(self.screen, (0, 0, 255), (int(pos[0] * 100), int(pos[1] * 100)), 10)
            pygame.draw.circle(self.screen, (0, 255, 0), (int(end_pos[0] * 100), int(end_pos[1] * 100)), 10)
            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(aircraft.start_position[0] * 100),
                                int(aircraft.start_position[1] * 100)), 10)

    def render(self):
        self.screen.fill((255, 255, 255))
        self.draw_aircraft()
        pygame.display.flip()

    def close(self):
        pygame.quit()
        sys.exit()
