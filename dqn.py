import loguru
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from loguru import logger


# Define the network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def print_parameters(self):
        for parameter in self.parameters():
            logger.debug(parameter)
            break


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (
            torch.tensor(np.array(states)).float().to(device),
            torch.tensor(np.array(actions)).long().to(device),
            torch.tensor(np.array(rewards)).unsqueeze(1).float().to(device),
            torch.tensor(np.array(next_states)).float().to(device),
            torch.tensor(np.array(dones)).unsqueeze(1).int().to(device)
        )

    def __len__(self):
        return len(self.buffer)


# Define the Vanilla DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, seed, learning_rate=1e-3, capacity=1000000,
                 discount_factor=0.99, update_every=4, batch_size=64, hidden_dim=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0
        logger.info(f'state_size: {state_size}, action_size: {action_size}, learning_rate: {learning_rate}, capacity: {capacity}, discount_factor: {discount_factor}, update_every: {update_every}, hidden_dim: {hidden_dim}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = QNetwork(
            state_size, action_size, hidden_dim).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),
                                    lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity)

    def step(self, state, action, reward, next_state, done, callback=None):
        # Save experience in replay buffer
        self.replay_buffer.push(state.flatten(), action, reward, next_state.flatten(), done)

        # Learn every update_every steps
        self.steps += 1
        if self.steps % self.update_every == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences, callback)

    def act(self, state, eps=0.0, explore_rule=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.tensor(state).float().flatten().to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy()), action_values.cpu().detach()
        elif explore_rule is None:
            return random.choice(np.arange(self.action_size)), np.zeros(self.state_size)
        else:
            better_chain_pos = explore_rule.explore()
            return better_chain_pos, np.zeros(self.state_size)

    def learn(self, experiences, callback=None):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from local model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.discount_factor * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.view(-1, 1))

        # self.qnetwork_local.print_parameters()
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # logger.debug(loss)
        if callback is not None:
            callback(loss)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % 50 == 0:
            self.qnetwork_target.load_state_dict(
                self.qnetwork_local.state_dict())
