import torch.nn as nn
import torch

class MLPDenoiser(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x, t):
        t_emb = t.unsqueeze(1)  # [batch, 1]
        x_in = torch.cat([x, t_emb], dim=1)  # [batch, 3]
        return self.net(x_in)
