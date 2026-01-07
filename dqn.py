import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn

        # -------- Shared layers --------
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 

        if self.enable_dueling_dqn:
            # -------- Value stream --------
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # -------- Advantage stream --------
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  

        if self.enable_dueling_dqn:
            # Value branch
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantage branch
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Combine value and advantage
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q


# -------------------- TEST --------------------
if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)

    state = torch.randn(10, state_dim)  
    output = net(state)

    print(output)
    print("Output shape:", output.shape)
