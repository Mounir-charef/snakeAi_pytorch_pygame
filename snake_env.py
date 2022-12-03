import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


point = namedtuple("Point", 'x, y')

BLOCK_SIZE = 10
SPEED = 144
# colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 128, 0)
BLACK = (20, 20, 20)


class SnakeEnv:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.frame_iteration = None
        self.food = None
        self.score = None
        self.head = None
        self.snake = None
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake game')
        self.clock = pygame.time.Clock()
        self.direction = None
        # init game state
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = point(self.w / 2, self.h / 2)
        self.snake = [self.head, point(self.head.x - BLOCK_SIZE, self.head.y),
                      point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        # 1. get input
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # 2. move the snake
        self._move(action)
        self.snake.insert(0, self.head)
        # 3. check if game is over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward -= 5
            return reward, game_over, self.score
        # 4. check if food was eaten
        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. refresh display
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. game over
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        # hits boundary
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        # [ straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = point(x, y)

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, YELLOW, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(f"Score is : {self.score}", True, WHITE)
        self.display.blit(text, [3, 0])
        pygame.display.flip()


if __name__ == '__main__':
    exit()
