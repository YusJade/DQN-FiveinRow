from loguru import logger
import numpy as np
import random


class ExploreRule:
    """
    Base class for exploration rules. Subclasses should implement the `explore` method.
    """
    def explore(self):
        """
        Determine the exploration action based on the board state and player.
        :param board: 2D numpy array representing the game board.
        :param player: Player identifier (e.g., 1 for current player).
        :return: Selected position as a tuple (x, y).
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ChainBasedExploreRule(ExploreRule):
    def __init__(self):
        super().__init__()
        
        
    def prepare(self, board, player):
        self.board = board
        self.player = player
    
    def explore(self):
        """
        Explore by selecting a position that maximizes chain potential.
        :param board: 2D numpy array representing the game board.
        :param player: Player identifier.
        :return: Selected position as a tuple (x, y).
        """
        def check_direction(board, x, y, dx, dy, player):
            count = 0
            for _ in range(4):  # check up to 4 spaces in a direction
                x, y = x + dx, y + dy
                if 0 <= x < board.shape[0] and 0 <= y < board.shape[1] and board[y, x] == player:
                    count += 1
                else:
                    break
            return count

        potential_positions = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i, j] == 0:
                    chains = 0
                    # Check all 8 possible directions
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
                    for dx, dy in directions:
                        chains += check_direction(self.board, j, i, dx, dy, self.player)
                    potential_positions[(i, j)] = chains
                else:
                    potential_positions[(i, j)] = 0

        positions, weights = zip(*potential_positions.items())
        weights = np.array(weights) + 1
        weights = np.power(weights, 15)
        # logger.info(weights)
        probabilities = weights / weights.sum()
        # logger.info(probabilities)
        selected_index = np.random.choice(len(positions), p=probabilities)
        # logger.info(f'choose chain position {positions[selected_index]}')
        row, col = positions[selected_index]
        board_size = self.board.shape[0]
        
        # logger.info(f'exploration weight of ({2},{2}) is {weights[12]}')
        return row * board_size + col


if __name__ == "__main__":
    board = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    player = "white"
    rule = ChainBasedExploreRule()
    rule.prepare(board, 1 if player == "white" else 2)
    action = rule.explore()
    row, col = action // board.shape[1], action % board.shape[1]
    logger.info(f'tend to explore ({row},{col}), action: {action}')
