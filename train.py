from datetime import datetime
import os
import sys
import time
from loguru import logger
import loguru
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import dqn
from explore_rule import ChainBasedExploreRule
from utils import render_history


def train():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    str_date = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    save_path = f"./runs/run_{str_date}"
    logger.add(f'./logs/{str_date}/train.log', level="DEBUG")
    os.makedirs(f"./runs/run_{str_date}", exist_ok=True)
    cfg = config.Config()
    agent = dqn.DQNAgent(
        cfg.state_size,
        cfg.action_size,
        cfg.seed,
        cfg.learning_rate,
        cfg.memory_capacity,
        cfg.discount_factor,
        batch_size=cfg.batch_size,
        hidden_dim=cfg.hidden_dim
    )

    import env
    env = env.FiveInRow(board_size=cfg.board_size)
    explore_rule = ChainBasedExploreRule()

    rewards = []    # 每局训练的总 reward
    avg_rewards = []     # 最近 50 局训练的平均 reward
    eps = cfg.eps_start

    for i_episode in range(cfg.num_episodes):
        env.reset()
        reward = 0
        episode_reward = 0
        eps = max(cfg.eps_end, cfg.eps_decay * eps)
        player = 'white'
        step = 0
        while True:
            step += 1
            state = env.get_state(player)
            explore_rule.prepare(env.get_board(), 1 if player == "white" else 2)
            action, _ = agent.act(state, eps, explore_rule)
            next_state, reward, done, invalid_action, trunc = env.step(
                player, action)
            agent.step(state, action, reward, next_state, done)
            episode_reward += reward
            if done or trunc:
                logger.info(f'done: {player if done else "null"}, '
                            f'trunc: {trunc}, step: {step}')
                break
            # 落子有效，转换黑白方
            if not invalid_action:
                player = "white" if player == "black" else "black"
            logger.debug(f'action: {action}, reward: {reward}')

        # if i_episode in [cfg.num_episodes * rate for rate in [0.1, 0.3, 0.5, 0.7, 0.9]]:
            # render_history(env.history)
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-50:]))
        logger.info(f'episode: {i_episode}, episode_reward:'
                    f'{episode_reward}, eps: {eps}')

    torch.save(agent.qnetwork_local.state_dict(), f'./runs/run_{str_date}/weight.pth')

    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.plot(range(len(rewards)), rewards, color="#4dfa1e")
    plt.plot(range(len(avg_rewards)), avg_rewards, color="#C51EFA")
    plt.legend(['reward', 'avg_reward'])
    plt.savefig(f'{save_path}/reward.png')
    plt.show()


if __name__ == "__main__":
    train()
