import numpy as np
from loguru import logger


def render_history(history):
    if len(history) == 0:
        return

    import pygame
    import sys
    from pygame.locals import QUIT, KEYDOWN, K_SPACE

    # 初始化 pygame
    pygame.init()
    cell_size = 30  # 每个格子的大小
    margin = 20     # 棋盘的边距
    board_size = np.size(history[0][0], 0)
    screen_size = board_size * cell_size + 2 * margin
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("过程回放：五子棋")

    # 定义颜色
    COLORS = {
        "background": (240, 240, 240),
        "grid": (0, 0, 0),
        "player1": (189, 245, 166),    # 黑棋
        "player2": (145, 232, 252),  # 红棋
        "empty": (240, 240, 240),
        "green": (189, 245, 166),
        "blue": (145, 232, 252),
        "black": (0, 0, 0),
    }
    COLORS["player1"] = COLORS["black"]
    COLORS["player2"] = COLORS["empty"]

    # 绘制单步棋盘
    def draw_board(board):
        screen.fill(COLORS["background"])
        for i in range(board_size + 1):
            # 绘制网格线
            pygame.draw.line(screen, COLORS["grid"],
                            (margin, margin + i * cell_size),
                            (margin + board_size * cell_size, margin + i * cell_size))
            pygame.draw.line(screen, COLORS["grid"],
                            (margin + i * cell_size, margin),
                            (margin + i * cell_size, margin + board_size * cell_size))
        # 绘制棋子
        for x in range(board_size):
            for y in range(board_size):
                if board[x, y] == 1:
                    pygame.draw.circle(screen, COLORS["black"],
                                    (margin + y * cell_size + cell_size // 2,
                                        margin + x * cell_size + cell_size // 2), cell_size // 2.5)
                    pygame.draw.circle(screen, COLORS["player1"],
                                    (margin + y * cell_size + cell_size // 2,
                                        margin + x * cell_size + cell_size // 2), cell_size // 3)
                elif board[x, y] == 2:
                    pygame.draw.circle(screen, COLORS["black"],
                                    (margin + y * cell_size + cell_size // 2,
                                        margin + x * cell_size + cell_size // 2), cell_size // 2.5)
                    pygame.draw.circle(screen, COLORS["player2"],
                                    (margin + y * cell_size + cell_size // 2,
                                        margin + x * cell_size + cell_size // 2), cell_size // 3)

    # 轮播对弈历史
    step = 0
    clock = pygame.time.Clock()
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:  # 按空格暂停/继续
                    paused = not paused

        if not paused:
            board, pos, msg = history[step]
            draw_board(board)
            pygame.display.flip()
            logger.info(f"step: {step}, pos: {pos}, msg: {msg}")
            step = (step + 1) % len(history)
            if step == 0:
                paused = True   # 对弈结束暂停播放
            clock.tick(30)   # 控制轮播速度（2帧每秒）

    pygame.quit()


def render_qvalue(qvalue, wid, hei):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    matrix = qvalue.reshape(wid, hei)

    # 定义自定义的 colormap：负数偏褐色，正数偏绿色
    colors = [(0.6, 0.3, 0.1), (1, 1, 1), (0.1, 0.8, 0.1)]  # 褐色-白色-绿色
    n_bins = 100  # 定义渐变的细腻程度
    cmap_name = 'custom_heatmap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # 显示热力图
    plt.imshow(matrix, cmap=custom_cmap, interpolation='nearest')
    plt.colorbar(label="Value")
    plt.title("Heatmap of Tensor")
    plt.show()
