import math
import stat
import sys
from loguru import logger
import loguru
import numpy as np

from chess_manual import ChessManual


class FiveInRow:
    def __init__(
            self,
            board_size=21,
            win_length=5,
            count_award=0.2,
            win_award=10
    ):
        self.board_size = board_size
        self.win_length = win_length
        self.white_board = np.zeros((board_size * board_size), dtype=np.int)
        self.black_board = np.zeros((board_size * board_size), dtype=np.int)
        self.board = np.zeros((board_size, board_size), dtype=np.int)
        self.history = []  # 缓存对弈过程
        self.manual = ChessManual()
        
    def reset(self):
        """
        重置环境，初始化棋盘。
        """
        self.board = np.zeros(
            (self.board_size, self.board_size), dtype=np.int)
        self.white_board = np.zeros((self.board_size * self.board_size),
                                    dtype=np.int)
        self.black_board = np.zeros((self.board_size * self.board_size),
                                    dtype=np.int)
        self.done = False
        self.winner = None
        self.history = []  # 清空缓存
        return self.board.copy()

    def step(self, player, action, callback=None):
        assert player in ["white", "black"], "illegal player!"
        row = action // self.board_size
        col = action % self.board_size
        if player == "white":
            player_board = self.white_board
            enemy_board = self.black_board
        elif player == "black":
            player_board = self.white_board
            enemy_board = self.black_board
        state_size = self.board_size * self.board_size * 2
        state = np.reshape([player_board, enemy_board], (1, state_size))
        # 棋盘上的目标位置已有棋子，当前动作为无效动作
        if self.board[row, col] != 0:
            return (state, 0.00001, self.done, True, False)
        # 计算执行动作前的优势值
        pre_reward = self.manual.get_reward(self.board,
                                            1 if player == "white" else 2)
        enemy_pre_reward = self.manual.get_reward(self.board,
                                             2 if player == "white" else 1)
        # 执行动作
        player_board[action] = 1
        state = np.reshape([player_board, enemy_board], (1, state_size))
        self.board[row, col] = 1 if player == "white" else 2
        self.done = self.check_winner(1 if player == "white" else 2, col, row)
        # 记录对弈历史，检查棋盘是否已满
        self.history.append((self.board.copy(), (col, row), f"put {player}"))
        trunc = np.sum(self.board == 0) == 0
        # 计算执行动作后的优势值
        next_reward = self.manual.get_reward(self.board,
                                             1 if player == "white" else 2)
        enemy_next_reward = self.manual.get_reward(self.board,
                                             2 if player == "white" else 1)
        enemy_deadline = (enemy_pre_reward - enemy_next_reward) * 0.0
        player_boost = (next_reward - pre_reward) * 1
        reward = player_boost + enemy_deadline
        return state, reward + 0.01, self.done, False, trunc

    def get_state(self, player):
        assert player in ["white", "black"], "illegal player!"
        if player == "white":
            player_board = self.white_board
            enemy_board = self.black_board
        elif player == "black":
            player_board = self.white_board
            enemy_board = self.black_board
        state_size = self.board_size * self.board_size * 2
        state = np.reshape([player_board, enemy_board], (1, state_size))
        return state
    
    def get_board(self):
        return self.board

    def check_winner(self, player, x, y):
        """
        检查给定位置是否使得当前玩家获胜。

        Args:
            player (int): 当前玩家 (1 或 -1)。
            x (int): 检查的行坐标。
            y (int): 检查的列坐标。

        Returns:
            bool: 是否胜利。
        """
        directions = [
            (1, 0),  # 水平
            (0, 1),  # 垂直
            (1, 1),  # 主对角线
            (1, -1)  # 副对角线
        ]

        for dx, dy in directions:
            count = 1
            for step in range(1, self.win_length):
                nx, ny = x + step * dx, y + step * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny, nx] == player:
                    count += 1
                else:
                    break

            for step in range(1, self.win_length):
                nx, ny = x - step * dx, y - step * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny, nx] == player:
                    count += 1
                else:
                    break

            if count >= self.win_length:
                return True

        return False