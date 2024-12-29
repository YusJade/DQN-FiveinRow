from loguru import logger
import numpy as np


class ChessManual():
    '''
    根据棋谱来评估棋盘的形势

    对棋盘的描述：
        *: 已有本方棋子
        +: 可落子的点位
        -: 已有对方棋子
    '''

    def __init__(self):
        self.manual = np.array([
            (['*', '*', '*', '*', '*'], 7000),
            # 四连子
            (['+', '*', '*', '*', '*', '+'], 5000),
            (['*', '*', '*', '*', '+', '-'], 1000),
            (['*', '*', '*', '*', '+', '+'], 2000),
            # 三连子
            (['*', '*', '*', '+', '+'], 800),
            (['+', '*', '*', '*', '+'], 800),
            (['+', '*', '*', '*', '-'], 400),
            # 二连子
            (['+', '*', '*', '+', '+'], 1000),
            (['+', '*', '*', '+', '+'], 80),
            (['+', '*', '*', '+', '-'], 60),
            (['+', '*', '*', '-', '-'], 40),
            # 一连子
            (['+', '*', '+', '+', '+'], 100),
            (['*', '+', '+', '+', '+'], 80),
            (['+', '+', '*', '+', '+'], 60),
        ], dtype=object)
        reverse_manual = np.array([(m[0][::-1], m[1]) for m in self.manual],
                                  dtype=object)
        self.manual = np.append(self.manual, reverse_manual, axis=0)
        self.reward_rules = self.manual[:, 0]

        logger.info(f'rules: {self.reward_rules}')

    def get_reward(self, board, player):
        board_trans = self.translate(board, player)
        num_row = board.shape[0]
        num_col = board.shape[1]

        total_reward = 0
        board_situations = []
        # 获取棋盘上的所有形势
        for row in range(num_row):
            for col in range(num_col):
                sit = self.get_all_situation(board_trans, row, col)
                board_situations += sit

        for sit in board_situations:
            for rule in self.reward_rules:
                if sit == rule:
                    total_reward += self.get_reward_by_rule(sit)

        return total_reward

    def get_reward_by_rule(self, rule):
        for m in self.manual:
            if rule == m[0]:
                return m[1]
        return 0

    def get_all_situation(self, board_trans, row, col):
        directions = [
            # 扫描六个位置为一个序列
            # 起始点偏右
            [(row, col + step) for step in range(-3, 3)],           # 横向
            [(row + step, col) for step in range(-3, 3)],           # 竖向
            [(row + step, col + step) for step in range(-3, 3)],    # 左斜向
            [(row + step, col - step) for step in range(-3, 3)],     # 右斜向
            # 起始点偏左
            [(row, col + step) for step in range(-2, 4)],           # 横向
            [(row + step, col) for step in range(-2, 4)],           # 竖向
            [(row + step, col + step) for step in range(-2, 4)],    # 左斜向
            [(row + step, col - step) for step in range(-2, 4)],    # 右斜向
            # 扫描五个位置为一个序列
            [(row, col + step) for step in range(-2, 3)],           # 横向
            [(row + step, col) for step in range(-2, 3)],           # 竖向
            [(row + step, col + step) for step in range(-2, 3)],    # 左斜向
            [(row + step, col - step) for step in range(-2, 3)],     # 右斜向
        ]

        result = []
        for direction in directions:
            elements = []
            for row, col in direction:
                # 确保索引在数组范围内
                if (0 <= row < board_trans.shape[0] and
                        0 <= col < board_trans.shape[1]):
                    elements.append(board_trans[row, col])
                else:
                    elements = []  # 如果超出范围，丢弃当前方向
                    break
            if elements:  # 如果有效的元素，加入结果
                result.append(elements)

        return result

    def translate(self, array, player):
        """
        将一个五子棋0-1-2棋盘按照下列规则转化
            - 0 -> '+'
            - 1 -> '+' if player == 1 else '-'
            - 2 -> '+' if player == 2 else '-'

        Parameters:
        - array (np.ndarray): A 2D numpy array to be translated.
        - player (int): The player number (1 or 2).

        Returns:
        - np.ndarray: A 2D numpy array with translated values.
        """
        if player not in [1, 2]:
            raise ValueError("player must be 1 or 2")

        # Create an output array of the same shape
        result = np.empty_like(array, dtype=str)

        # Apply the translation rules
        result[array == 0] = '+'
        result[array == 1] = '*' if player == 1 else '-'
        result[array == 2] = '*' if player == 2 else '-'

        return result


if __name__ == "__main__":
    manual = ChessManual()
    logger.info(manual.manual)

    test_board = np.array([
        [1, 2, 0, 0, 1, 0],
        [0, 2, 0, 0, 2, 1],
        [1, 0, 2, 0, 0, 2],
        [1, 0, 2, 2, 1, 0],
        [1, 0, 2, 0, 1, 0],
        [1, 0, 2, 0, 1, 0],
    ])
    target_player = 2
    board_trans = manual.translate(test_board, target_player)
    logger.info(f'translated board: (target_player: {target_player})'
                f'\n{board_trans}')
    row = 2
    col = 2
    situations = manual.get_all_situation(board_trans, row, col)
    logger.info(f'situations on (row: {row}, col: {col}): \n {situations}')

    reward = manual.get_reward(test_board, target_player)
    logger.info(f'reward: {reward}')