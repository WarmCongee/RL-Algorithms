from collections import deque

import gym
import torch
import torch.nn as nn  # 各种层类型的实现
import torch.nn.functional as F  # 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim  # 实现各种优化算法的包
from torchvision import datasets
import random

# @Author  : WarmCongee
# @Function: classic RL-Algorithms module

GAMMA = 0.9
LEARN_RATE = 0.01
EPSILON_STRATEGY = 0.0
BATCH_SIZE = 32  # 一次训练的样本数目，提高GPU利用率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


class Network(nn.Module):
    def __init__(self, input_params, output_params):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_params, 30)  # 全连接层，稀疏连接似乎需要自己实现
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, output_params)

    def forward(self, input_value):  # 会自动实现反向函数
        out = F.relu(self.fc1(input_value))
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def init_weights_bias(self):
        for item in self.modules():
            nn.init.normal_(item.weight.data, 0, 0.2)  # 初始化权重
            item.bias.data.zero_()  # 初始化ｄ偏移量


class Agent:
    def __init__(self, env):
        self.state_params = env.observation_space.shape[0]  # 获取状态空间第一个维度的数目 用于后续实例化神经网络时初始化输入层个数
        self.action_params = env.action_space.n  # 动作个数

        self.network = Network(self.state_params, self.action_params)
        self.network.init_weights_bias()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARN_RATE)
        self.loss_function = nn.MSELoss(reduction='none')

        self.replay_buffer = deque()  # 以原始的队列作为经验回访池
        self.epsilon = EPSILON_STRATEGY

    def train_network(self):
        self.time_step += 1

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)  # 32的list
        state_batch = torch.FloatTensor([data[0] for data in minibatch]).to(device)  # 32*4
        action_batch = torch.LongTensor([data[1] for data in minibatch]).to(device)  # 32*2
        reward_batch = torch.FloatTensor([data[2] for data in minibatch]).to(device)  # 32*1
        next_state_batch = torch.FloatTensor([data[3] for data in minibatch]).to(device)  # 32*4
        done = torch.FloatTensor([data[4] for data in minibatch]).to(device)

        done = done.unsqueeze(1)
        reward_batch = reward_batch.unsqueeze(1)
        # q_val = self.network.forward(state_batch)  # 32*2
        action_index = action_batch.argmax(dim=1).unsqueeze(1)  # 32*1
        eval_q = self.network.forward(state_batch).gather(1, action_index)  # 32*1

        # Step 2: calculate y
        Q_value_batch = self.network.forward(next_state_batch)
        next_action_batch = torch.unsqueeze(torch.max(Q_value_batch, 1)[1], 1)
        next_q = self.network.forward(next_state_batch).gather(1, next_action_batch)

        y_batch = reward_batch + GAMMA * next_q * (1 - done)
        # y_batch = torch.tensor(y_batch).unsqueeze(1)

        # 更新网络
        loss = self.loss_function(eval_q, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epsilon_greedy(self, state):  # ε-greedy策略选择
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)  # 给state加一个batch_size的维度，此时batch_size为1
        Q_value = self.network.forward(state.to(device))

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_params - 1)
        else:
            return torch.max(Q_value, 1)[1].data.to('cpu').numpy()[0]




