import torch.nn as nn
import torch

class ResidualMLPDenoiser(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(input_dim + 1, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x, t):
        t_emb = t.unsqueeze(1)  
        x_in = torch.cat([x, t_emb], dim=1)
        h = self.activation(self.input_layer(x_in))

        h1 = self.activation(self.hidden1(h))
        h = h + h1  

        h2 = self.activation(self.hidden2(h))
        h = h + h2  

        return self.output_layer(h)
