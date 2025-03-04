import torch
import torch.nn as nn
import torch.nn.functional as F

# Towards target position
# Input dim (14) = direction (3) + distance (1) + attitude (4) + qd (6)
class Towards(nn.Module):
    def __init__(self, input_dim=14, output_dim=4):
        super(Towards, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        
        self.output_layer = nn.Linear(32, output_dim)
        
    def forward(self, dir, dist, att, qd):
        x = torch.cat((dir, dist, att, qd), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))     
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        output = torch.sigmoid(self.output_layer(x))
        
        return output