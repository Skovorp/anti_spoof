import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Optional, Any
from tqdm import tqdm
import os
import wandb
from copy import copy

from dataset import ASVspoofDataset
from dataset import collate_fn as custom_collate_fn
from model import RawNet2
from utils import pretty_now, eer_metric
from train import validation_epoch


if __name__ == "__main__":
    config_path = '/home/ubuntu/anti_spoof/config.yaml'
    model_path = '/home/ubuntu/anti_spoof/saved/2023-12-17_19:55:44.pth'
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)


    # data
    test_set = ASVspoofDataset(**cfg['dataset']['test'])
    test_loader = DataLoader(test_set, collate_fn=custom_collate_fn, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=8)
    
    print("len test set", len(test_set))
    print("len test loader", len(test_loader))

    # model
    device = torch.device('cuda')
    model = RawNet2(**cfg['model']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(model)
    
    class_weights = torch.FloatTensor(eval(cfg['training']['ce_weights'])).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss, eer = validation_epoch(model, criterion, test_loader, 'Loading sota model...')

    print(f"Test loss: {loss}")
    print(f"Test EER:  {100 * eer:.2f}%")
    print("ПОБЕДА!!!")
    