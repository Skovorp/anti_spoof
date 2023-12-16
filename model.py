import torch
from torch import nn

class RawNet2(nn.Module):
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