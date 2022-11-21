import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


point = namedtuple("Point", 'x, y')

BLOCK_SIZE = 20
SPEED = 15
# colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (20, 20, 200)
BLUE2 = (0, 100, 255)
BLACK = (20, 20, 20)


class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake game')
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT
        self.head = point(self.w / 2, self.h / 2)
        self.snake = [self.head, point(self.head.x - BLOCK_SIZE, self.head.y),
                      point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        # 1. get input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        # 2. move the snake
        self._move(self.direction)
        self.snake.insert(0, self.head)
        # 3. check if game is over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
        # 4. check if food was eaten
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. refresh display
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. game over
        return game_over, self.score

    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True

        return False

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = point(x, y)

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(f"Score is : {self.score}", True, WHITE)
        self.display.blit(text, [3, 0])
        pygame.display.flip()


def main():
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    pygame.quit()
    print('Your score :', score)


if __name__ == '__main__':
    main()
