import random
from dataclasses import dataclass
from typing import Literal


@dataclass
class Edge:
    kind: Literal['top', 'bottom', 'left', 'right']
    env_width: float
    env_height: float
    seed: int

    def __post_init__(self):
        random.seed(self.seed)

        if self.kind == 'top':
            self.x, self.y = random.uniform(0, self.env_width), self.env_height
        elif self.kind == 'bottom':
            self.x, self.y = random.uniform(0, self.env_width), 0
        elif self.kind == 'left':
            self.x, self.y = 0., random.uniform(0, self.env_height)
        elif self.kind == 'right':
            self.x, self.y = self.env_width, random.uniform(0, self.env_height)
