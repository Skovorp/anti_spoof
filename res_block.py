import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels != out_channels:
            self.resample = nn.Conv1d(in_channels,out_channels,kernel_size=1)
        else:
            self.resample = nn.Identity()

        self.pipeline = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.reduce_pool = nn.MaxPool1d(kernel_size=3)

        self.fms_avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.fms_linear = nn.Linear(in_features=out_channels, out_features=out_channels)

        
    def forward(self, x):
        res = self.pipeline(x) + self.resample(x)
        res = self.reduce_pool(res)
        
        t = self.fms_avg(res).squeeze(2) # (n, c, t) -> (n, c, 1) -> (n, c)
        t = F.sigmoid(self.fms_linear(t)).unsqueeze(2) # (n, c) -> (n, c, 1)  channels are from 0 to 1
        res = res * t + t
        return res
