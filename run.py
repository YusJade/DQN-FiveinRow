import random
import torch
from loguru import logger
import numpy as np

from config import Config
from dqn import QNetwork
from env import FiveInRow
from utils import render_history

cfg = Config()

net = QNetwork(cfg.state_size, cfg.action_size, cfg.hidden_dim)
net.load_state_dict(torch.load("runs/run_2025-01-03_15_49_01/weight.pth", weights_only=True))
net.eval()
logger.info(net)

env = FiveInRow(board_size=cfg.board_size)
state = env.reset()
invalid_action = False
player = 'white'
while True:
    state = env.get_state(player)
    input_tensor = torch.tensor(state).float().flatten()
    if invalid_action:
        action = random.choice(range(cfg.action_size))
    else:
        with torch.no_grad():
            output_tensor = net.forward(input_tensor).cpu().detach()
        action = np.argmax(output_tensor.numpy())
    next_state, _, done, invalid_action, trunc = env.step(player, action)
    if done or trunc:
        break
    if not invalid_action:
        player = "white" if player == "black" else "black"
    logger.info(f"action: {action}(col: {action % env.board_size}, row: {action // env.board_size}), invalid: {invalid_action}")

render_history(env.history)

