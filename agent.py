import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- CONFIGURATION FROM PAPER TABLE 2 ---
GAMMA = 0.9
BATCH_SIZE = 256
BUFFER_SIZE = 576  # Paper: "stores last 576 values (24 days)"
LR_START = 0.8     # Paper: aggressive start
LR_FINAL = 0.01
EPS_START = 1.0
EPS_FINAL = 0.05
TARGET_UPDATE_FREQ = 2 

class DDQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DDQN, self).__init__()
        # Paper 4.2: "fully connected... two hidden layers... layer normalization"
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),  # Critical paper requirement
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, input_dim=10, action_dim=11):
        self.action_dim = action_dim
        self.input_dim = input_dim
        
        # Networks
        self.policy_net = DDQN(input_dim, action_dim).float()
        self.target_net = DDQN(input_dim, action_dim).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001) # Adam default, schedulers handled in main
        self.memory = deque(maxlen=BUFFER_SIZE)
        
        self.epsilon = EPS_START
        self.steps_done = 0

    def get_action(self, state, eval_mode=False):
        # Epsilon-Greedy Strategy
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # --- DDQN Logic ---
        # 1. Select action using Policy Net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            # 2. Evaluate value using Target Net
            next_q_values = self.target_net(next_states).gather(1, next_actions)

        target_q = rewards + (GAMMA * next_q_values * (1 - dones))
        current_q = self.policy_net(states).gather(1, actions)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def copy_weights_from(self, other_agent):
        """Used for Parameter Sharing phase"""
        self.policy_net.load_state_dict(other_agent.policy_net.state_dict())
        self.target_net.load_state_dict(other_agent.target_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())