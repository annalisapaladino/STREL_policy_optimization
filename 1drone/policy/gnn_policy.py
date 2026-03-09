import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class DroneGNNPolicy(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=2, action_scale=2.0):
        super().__init__()
        self.action_scale = float(action_scale)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
        )

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        drone_mask = x[:, 5] == 1.0
        h_drones = h[drone_mask]
        actions = torch.tanh(self.fc(h_drones)) * self.action_scale
        return actions
