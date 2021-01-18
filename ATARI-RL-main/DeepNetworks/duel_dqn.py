import torch
import torch.nn as nn 

class D_DQN(nn.Module):

      def __init__(self, obs_size, n_actions):
        super(D_DQN, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()

        self.fc2_A = nn.Linear(64, 256)
        self.fc3_A = nn.Linear(256, n_actions)
        
        self.fc2_V = nn.Linear(64, 256)
        self.fc3_V = nn.Linear(256, 1)
        

      def forward(self, state):
        # ====================================================
        
        x = self.relu(self.fc1(state))
        
        x_A = self.relu(self.fc2_A(x))
        A = self.fc3_A(x_A)

        x_V = self.relu(self.fc2_V(x))
        V = self.fc3_V((x_V))

        Q = V + A - torch.mean(A,dim=1).unsqueeze(1)

        return Q
