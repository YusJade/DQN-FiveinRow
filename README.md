# DQN-FiveinRow

这是一个使用 DQN 强化学习实现的五子棋 AI，整个工程由 基于 Web 的部署展示部分和基于 Pytorch 的算法实现部分。

## 基于 Pytorch 的算法实现部分

- `config.py`: `train.py`、`run.py`的相关配置。
- `train.py`: 训练 DQN 网络，采用经验回放和 Double-DQN 方法。
- `run.py`: 加载训练后的网络权重并执行推理。
- `export.py`: 将模型导出为可以在浏览器本地部署的 onnx 模型。

- `dqn.py`: 定义 DQN 网络的结构以及实现 agent 智能体。
- `env.py`: 实现五子棋的游戏环境。
- `chess_manual.py`: 棋谱定义了可以得分的情况，用以计算 reward。

**经验回放（Experience Replay）**：为了打破数据之间的相关性并提高学习的效率，DQN会将智能体的经验（状态、动作、奖励、新状态）存储在一个数据集中，然后从中随机抽取样本进行学习。

**目标网络（Target Network）**：DQN 使用了两个神经网络，一个是在线网络，用于选择动作；一个是目标网络，用于计算TD目标（Temporal-Difference Target）。这两个网络有相同的结构，但参数不同。在每一步学习过程中，我们使用在线网络的参数来更新目标网络的参数，但是更新的幅度较小。这样可以提高学习的稳定性。

## 基于 Web 的部署展示部分

- `dqn-fiveinrow-web/src/ai/index.ts`: 使用 onnxruntime-web 将 onnx 模型部署到浏览器本地。

## 不足

AI 的智能过低，无法形成连子和阻碍对手连子，可能是因为棋盘状态过于复杂，AI 无法遇到并学习所有局面下的最优决策，可能通过改进探索策略来解决。