import torch
import torch.nn as nn
import torch.nn.functional as F


class myMLP(nn.Module):
    def __init__(self, feature_dim_o=1984, feature_dim=128):
        super(myMLP, self).__init__()

        self.fcUnit = feature_dim_o
        self.m1 = self.fcUnit//2
        self.m2 = self.m1//2
        # projection head
        self.g = nn.Sequential(
            nn.Linear(self.fcUnit, self.m1, bias=False),
            nn.PReLU(),
            nn.Linear(self.m1, feature_dim, bias=False))

    def forward(self, x):
        out = self.g(x)
        return F.normalize(out, dim=-1)




