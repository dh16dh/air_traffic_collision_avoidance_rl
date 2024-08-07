import sys
import numpy as np
import pygame


class PygameVisualizer:
    resize_multiplier = 2

    def __init__(self, env):
        pygame.init()
        self.env = env
        self.width = env.env_width * self.resize_multiplier
        self.height = env.env_height * self.resize_multiplier
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Aircraft Navigation")
        self.clock = pygame.time.Clock()

    def draw_aircraft(self):
        for aircraft in self.env.agents:
            pos = aircraft.position
            end_pos = aircraft.end_position
            start_pos = aircraft.start_position
            pygame.draw.circle(self.screen, (0, 255, 255), (int(pos[0] * self.resize_multiplier), int(self.height - pos[1] * self.resize_multiplier)), aircraft.PAZ * self.resize_multiplier)
            pygame.draw.circle(self.screen, (0, 0, 255), (int(pos[0] * self.resize_multiplier), int(self.height - pos[1] * self.resize_multiplier)), self.resize_multiplier)
            pygame.draw.circle(self.screen, (0, 255, 0), (int(end_pos[0] * self.resize_multiplier), int(self.height - end_pos[1] * self.resize_multiplier)), self.resize_multiplier)
            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(start_pos[0] * self.resize_multiplier),
                                int(self.height - start_pos[1] * self.resize_multiplier)), self.resize_multiplier)

    def render(self):
        self.screen.fill((255, 255, 255))
        self.draw_aircraft()
        pygame.display.flip()

    def close(self):
        pygame.quit()
        sys.exit()

    @staticmethod
    def check_for_end():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
