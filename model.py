import torch
from torch import nn
import torch.nn.functional as F

from res_block import ResBlock
from sinc_conv import SincConv_fast

class Baseline(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


        self.conv = nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=256,
                stride=128
            ),
            nn.ReLU()
        )

        self.recurent = nn.GRU(
            input_size=64,
            hidden_size=256,
            num_layers=1, 
            batch_first=True
        )
        self.linear_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        # print(x.shape)
        # print("before rec  ", f"{torch.cuda.memory_allocated():019_d}")
        x, _ = self.recurent(x)
        # print("after  rec  ", f"{torch.cuda.memory_allocated():019_d}")
        x = x[:, -1, :]
        x = self.linear_block(x)
        return x

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.size())).item() for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params:_}"
    
    
class RawNet2(nn.Module):
    def __init__(self, p_sinc, p_1res_block, p_2res_block, p_gru, **kwargs):
        super().__init__()

        self.sinc_filter = SincConv_fast(
                in_channels=1,
                out_channels=p_sinc['out_channels'],
                kernel_size=p_sinc['kernel_size'],
                min_low_hz=0,
                min_band_hz=0
            )
        self.sinc_max_pool = nn.MaxPool1d(kernel_size=3)

        self.first_res_block = nn.Sequential(
                ResBlock(p_sinc['out_channels'], p_1res_block['channels']),
                *[ResBlock(p_1res_block['channels'], p_1res_block['channels']) for _ in range(p_1res_block['count'] - 1)]
            )
        
        self.second_res_block = nn.Sequential(
                ResBlock(p_1res_block['channels'], p_2res_block['channels']),
                *[ResBlock(p_2res_block['channels'], p_2res_block['channels']) for _ in range(p_2res_block['count'] - 1)]
            )

        self.pre_gru_processing = nn.Sequential(
            nn.BatchNorm1d(num_features=p_2res_block['channels']),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        self.gru_layer = nn.GRU(
            input_size=p_2res_block['channels'],
            hidden_size=p_gru['hidden_size'],
            batch_first=True,
            num_layers=p_gru['num_layers']
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(p_gru['hidden_size'], p_gru['hidden_size']),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(p_gru['hidden_size'], 2),
        )
        
    def forward(self, x):
        x = self.sinc_filter(x)
        x = self.sinc_max_pool(torch.abs(x))
        x = self.first_res_block(x)
        x = self.second_res_block(x)
        x = self.pre_gru_processing(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru_layer(x)
        x = x[:, -1, :]
        x = self.mlp(x)
        return x
    
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.size())).item() for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params:_}"