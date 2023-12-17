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

from dotenv import load_dotenv
load_dotenv()

@torch.no_grad()
def validation_epoch(model, criterion: nn.Module, loader: DataLoader, tqdm_desc: str):
    val_loss = 0.0
    device = next(model.parameters()).device
    all_logits = []
    all_targets = []

    model.eval()
    for audios, targets in tqdm(loader, desc=tqdm_desc):
        audios = audios.to(device)
        targets = targets.to(device)
        logits = model(audios)
        val_loss += criterion(logits, targets).item() * audios.shape[0]
        all_logits.append(logits.cpu().detach())
        all_targets.append(targets.cpu().detach())
    val_loss /= len(loader.dataset)
    eer = eer_metric(torch.cat(all_logits, 0), torch.cat(all_targets, 0))
    return val_loss, eer


def training_epoch(model, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, epoch_size: int, tqdm_desc: str):
    device = next(model.parameters()).device
    train_loss = 0.0
    seen_objects = 0
    model.train()
    for step_num, (audios, targets) in tqdm(enumerate(loader), total=min(epoch_size, len(loader))):
        if step_num == epoch_size:
            break
        optimizer.zero_grad()
        audios = audios.to(device)
        targets = targets.to(device)
        # print("added to dev", f"{torch.cuda.memory_allocated():019_d}")
        logits = model(audios)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * audios.shape[0]
        wandb.log({'batch_train_loss': loss.item()})
        seen_objects += audios.shape[0]
    train_loss /= seen_objects
    return train_loss


def train(model, optimizer: torch.optim.Optimizer, criterion,
          train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, save_path: str, num_epochs: int, cfg):    
    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader, cfg['training']['epoch_size'],
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        print(f"Epoch {epoch}/{num_epochs}. train_loss: {train_loss}")# train_eer: {train_eer}")
        wandb.log({
            "epoch": epoch, 
            "epoch_train_loss": train_loss
        })

        val_loss, val_eer = validation_epoch(model, criterion, val_loader, tqdm_desc=f'Validation {epoch}/{num_epochs}')
        wandb.log({
            'val_loss': val_loss,
            'val_eer': val_eer
        })
        print(f"Epoch {epoch}/{num_epochs}. val_loss: {val_loss} val_eer: {val_eer}")
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
    print("Completed training. Running on test")
    test_loss, test_eer = validation_epoch(model, criterion, test_loader, tqdm_desc=f'Testing...')
    wandb.log({
            'test_loss': test_loss,
            'test_eer': test_eer
        })
    print("All done!")


if __name__ == "__main__":
    SEED = 123
    torch.manual_seed(SEED)


    config_path = '/home/ubuntu/anti_spoof/config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['training']['save_path'] = cfg['training']['save_path'].replace('{pretty_time}', pretty_now())
    cfg['training']['epoch_size'] = int(1e8) if not cfg['training']['epoch_size'] else cfg['training']['epoch_size']

    wandb.init(
        project="anti_spoof",
        config=cfg,
        # mode='disabled'
    )

    # data
    train_set = ASVspoofDataset(**cfg['dataset']['train'])
    val_set = ASVspoofDataset(**cfg['dataset']['val'])
    test_set = ASVspoofDataset(**cfg['dataset']['test'])
    train_loader = DataLoader(train_set, collate_fn=custom_collate_fn, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, collate_fn=custom_collate_fn, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, collate_fn=custom_collate_fn, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=8)
    
    print("len train set", len(train_set))
    print("len val set", len(val_set))
    print("len test set", len(test_loader))
    
    print("len train loader", len(train_loader))
    print("len val loader", len(val_loader))
    print("len test loader", len(test_loader))

    # model
    device = torch.device('cuda')
    model = RawNet2(**cfg['model']).to(device)
    print(model)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    class_weights = torch.FloatTensor(eval(cfg['training']['ce_weights'])).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train(model, optimizer, criterion, train_loader, val_loader, test_loader, cfg['training']['save_path'], cfg['training']['num_epochs'], cfg)
    wandb.finish()