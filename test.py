import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Optional, Any
from tqdm import tqdm
import os
import wandb
from copy import copy
import torchaudio

from dataset import ASVspoofDataset
from dataset import collate_fn as custom_collate_fn
from model import RawNet2
from utils import pretty_now, eer_metric
from train import validation_epoch


if __name__ == "__main__":
    config_path = '/home/ubuntu/anti_spoof/config.yaml'
    model_path = '/home/ubuntu/anti_spoof/saved/2023-12-17_19:55:44.pth'
    test_audio_dir = '/home/ubuntu/anti_spoof/test_audios'
    
    audio_paths = sorted(os.listdir(test_audio_dir))
    audio_paths = [(x.split('.')[0], test_audio_dir + '/' + x) for x in audio_paths]
    # print(f"Infering on {len(audio_paths)}: {[x[0] for x in audio_paths]}")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)


    # model
    device = torch.device('cuda')
    model = RawNet2(**cfg['model']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Probability spoof:")
    for name, path in audio_paths:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.unsqueeze(1).to(device)
        waveform = waveform[:1, :1, :] # only one channel
        logits = model(waveform)
        res = torch.softmax(logits, -1)
        print(f"{name:15}: {res[0][0].item() * 100:.4f}% logits: {logits.tolist()}")
    