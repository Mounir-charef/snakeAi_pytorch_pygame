import torch
import random
import numpy as np
from collections import deque
from snake_env import SnakeEnv, Direction, point, BLOCK_SIZE
from model import LinearQnet, QTrainer
import os
# from helper import plot
# from threading import Thread

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # random
        self.gamma = 0.9  # discount
        self.memory = deque(maxlen=MAX_MEM)
        self.model = LinearQnet(11, 512, 3)
        if os.path.exists('./models/model.pth'):
            checkpoint = torch.load('./models/model.pth')
            self.model.load_state_dict(checkpoint)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    @staticmethod
    def get_state(game):
        head = game.snake[0]
        point_l = point(head.x - BLOCK_SIZE, head.y)
        point_r = point(head.x + BLOCK_SIZE, head.y)
        point_u = point(head.x, head.y - BLOCK_SIZE)
        point_d = point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # def get_action(self, state):
    #     self.epsilon = 80 - self.n_games
    #     final_move = [0, 0, 0]
    #     if random.randint(0, 200) < self.epsilon:
    #         move = random.randint(0, 2)
    #         final_move[move] = 1
    #     else:
    #         state0 = torch.tensor(state, dtype=torch.float)
    #         prediction = self.model(state0)
    #         move = torch.argmax(prediction).item()
    #         final_move[move] = 1
    #
    #     return final_move

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


# def get_new_thread(func, *args):
#     thread = Thread(target=func, args=args)
#     thread.start()
#     thread.join()


def train():
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    record = 0
    agent = Agent()
    game = SnakeEnv()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train and plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # plot_scores.append(score)
            # total_score += score
            # plot_mean_scores.append(total_score / agent.n_games)
            # plot(plot_scores, plot_mean_scores)
            # get_new_thread(plot, (plot_scores, plot_mean_scores))


if __name__ == '__main__':
    train()
