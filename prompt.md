你是一个精通强化学习的AI助手，现在我需要你的建议。
这是一个使用 DQN（Deep Q-Learning）实现的五子棋 AI，其激励函数的设计如下：
>* 表示我方棋子， + 可落子的位置， - 表示对方棋子
```python
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
            (['+', '*', '*', '*', '*', '+'], 500),
            (['*', '*', '*', '*', '+', '-'], 100),
            (['*', '*', '*', '*', '+', '+'], 200),
            # 三连子
            (['*', '*', '*', '+', '+'], 800),
            (['+', '*', '*', '*', '+'], 800),
            (['+', '*', '*', '*', '-'], 40),
            # 二连子
            (['+', '*', '*', '+', '+'], 1000),
            (['+', '*', '*', '+', '+'], 800),
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
```
训练过程中，模型每次做出的行为所获得的 reward 会通过棋盘双方的力量值计算得出：manual 是一个棋谱，实现了激励函数的功能，它会检查被输入棋盘上的每个位置，每次检查都以当前位置为起始点，按照水平方向、竖直方向、左斜线方向和右斜线方向扫描临近的位置，每次扫描得到一个长度 5 到 11 的序列，用以记录一方在棋盘相应方位的落子情况，如果这些序列能在棋谱（manual）中检索到，则给予棋谱中设定的 reward。期望能通过这种设计，鼓励模型朝着尽可能形成多个方向的连子的趋势落子，从而强化自身的智能。
目前模型的智能仍旧很弱，希望通过语言大模型的能力，优化激励函数：

基本概念：
  - 输入状态： 一个大小 242 的一维张量，前 121 表示我方在 11x11 棋盘的落子情况，后 11x11 表示对方在 11x11棋盘的落子情况，0 表示未落子，1 表示已落子。
  - 输出状态： 一个大小 121 的一维张量，表示在 11x11 棋盘的各个位置落子的 Q 值。
  
我会提供给你形如 0 21 32 24 12 ...的序列，它代表一局五子棋的对弈过程，默认为玩家先手， AI 后手，数字代表落子在棋盘上的位置。现在需要你分析对弈过程，给出你对激励函数的调整以及理由，调整后的激励函数以以下形式呈现：
```python
[
    (['*', '*', '*', '*', '*'], 10000, "五连子（提高奖励）"),
    (['+', '*', '*', '*', '*', '+'], 1000, "四连子（两端可落子，提高奖励）"),
    (['*', '*', '*', '*', '+', '-'], 200, "四连子（一端被阻断，提高奖励）"), 
    (['*', '*', '*', '*', '+', '+'], 500, "四连子（一端可落子，提高奖励）"),
    (['-', '-', '-', '-', '+'], -1000, "对方四连子（一端可落子，提高惩罚）"),
    (['*', '*', '*', '+', '+'], 1500, "三连子（提高奖励）"),
    (['+', '*', '*', '*', '+'], 1500, "三连子（两端可落子，提高奖励）"),
    (['+', '*', '*', '*', '-'], 100, "三连子（一端被阻断，提高奖励）"),
    (['-', '-', '-', '+'], -500, "对方三连子（一端可落子，提高惩罚）"),
    (['+', '*', '*', '+', '+'], 1200, "二连子（提高奖励）"),

]
```