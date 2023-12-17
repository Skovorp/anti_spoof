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
from train import train
import traceback


if __name__ == "__main__":
    SEED = 123
    torch.manual_seed(SEED)
    
    ablation_config_path = '/home/ubuntu/anti_spoof/ablation_configs/'
    config_list = os.listdir(ablation_config_path) 
    print(config_list)
    trials = []
    some_config = ''
    for config_filename in config_list:
        name = config_filename.split('.')[0]
        with open(ablation_config_path + '/' + config_filename) as f:
            cfg = yaml.safe_load(f)
            cfg['training']['epoch_size'] = int(1e8) if not cfg['training']['epoch_size'] else cfg['training']['epoch_size']
            some_config = cfg
        trials.append((name, cfg))
    print([x[0] for x in trials])
    print(f"About to run {len(trials)} trials. ok?")
    input()
    print("hope you said yes...")
    
    # data
    train_set = ASVspoofDataset(**some_config['dataset']['train'])
    val_set = ASVspoofDataset(**some_config['dataset']['val'])
    test_set = ASVspoofDataset(**some_config['dataset']['test'])
    train_loader = DataLoader(train_set, collate_fn=custom_collate_fn, batch_size=some_config['training']['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, collate_fn=custom_collate_fn, batch_size=some_config['training']['batch_size'], shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, collate_fn=custom_collate_fn, batch_size=some_config['training']['batch_size'], shuffle=False, num_workers=8)
    
    print("len train set", len(train_set))
    print("len val set", len(val_set))
    print("len test set", len(test_loader))
    
    print("len train loader", len(train_loader))
    print("len val loader", len(val_loader))
    print("len test loader", len(test_loader))
    
    
    for trial_num, (trial_name, trial_cfg) in enumerate(trials):
        try:
            wandb.init(
                project="anti_spoof", 
                config=trial_cfg, 
                name=trial_name,
                # mode='disabled'
            )   
            device = torch.device('cuda')
            model = RawNet2(**trial_cfg['model']).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=trial_cfg['training']['lr'], weight_decay=trial_cfg['training']['weight_decay'])
            class_weights = torch.FloatTensor(eval(trial_cfg['training']['ce_weights'])).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            train(model, optimizer, criterion, train_loader, val_loader, test_loader, None, trial_cfg['training']['num_epochs'], trial_cfg)
            wandb.finish()
            print(f"Done {trial_num + 1}/{len(trials)}")
        except Exception:
            traceback.print_exc()
            print("FAILED!FAILED!FAILED!FAILED!FAILED!FAILED!FAILED!FAILED!FAILED!FAILED!FAILED!FAILED!FAILED!")
            print(f"Skipping trial {trial_num + 1} {trial_name}")
           
        