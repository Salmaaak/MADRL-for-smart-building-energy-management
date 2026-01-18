import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        # 2 Hidden Layers as per Section 4.2
        # Layer Normalization included as it reduces training time
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SharedAgent:
    def __init__(self, input_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Shared Networks (Parameter Sharing)
        self.policy_net = DQNetwork(input_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(input_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = [] 
        self.batch_size = 256
        self.gamma = 0.9 #
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.action_dim = action_dim

    def select_actions(self, obs_dict, eval_mode=False):
        """
        Takes dict of {zone: state}
        Returns dict of {zone: action}
        """
        actions = {}
        # We can batch process these for speed
        states = []
        zones = []
        
        for z, s in obs_dict.items():
            zones.append(z)
            states.append(s)
            
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        
        # Epsilon Greedy
        if not eval_mode and random.random() < self.epsilon:
            random_acts = [random.randint(0, self.action_dim - 1) for _ in zones]
            return dict(zip(zones, random_acts))

        with torch.no_grad():
            q_values = self.policy_net(states_t)
            best_acts = q_values.argmax(dim=1).cpu().numpy()
            return dict(zip(zones, best_acts))

    def store_transition(self, obs_dict, acts_dict, reward, next_obs_dict, done):
        # We store transitions per agent (zone) because the network learns "State -> Action" mapping
        # However, the reward is SHARED (Global). This is "Independent Q-Learning"
        for z in obs_dict:
            s = obs_dict[z]
            a = acts_dict[z]
            ns = next_obs_dict[z]
            # Store tuple
            self.memory.append((s, a, reward, ns, done))
        
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size: return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # DDQN Logic
        next_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states_t).gather(1, next_actions)
        
        expected_q = rewards_t + (self.gamma * next_q_values * (1 - dones_t))
        current_q = self.policy_net(states_t).gather(1, actions_t)

        loss = nn.MSELoss()(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())