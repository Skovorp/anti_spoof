import pandas as pd
import numpy as np
import torchaudio
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SEED = 228

class ASVspoofDataset(Dataset):
    def __init__(self, dataset_path, part, max_len=64000, limit=None):
        super().__init__()
        assert part in ['train', 'eval', 'dev'], "Part should be 'train', 'eval', or 'dev'"
        
        self.part = part
        self.max_len = max_len
        self.audio_paths, self.dataset_labels = self._load_dataset(dataset_path, part)
        
        if limit is not None:
            self.audio_paths = self.audio_paths[:limit]
            self.dataset_labels = self.dataset_labels[:limit]

    def _load_dataset(self, dataset_path, part):
        base_path = f'{dataset_path}/LA/LA'
        protocol_suffix = 'trn' if part == 'train' else 'trl'
        protocol_file = f'ASVspoof2019.LA.cm.{part}.{protocol_suffix}.txt'
        LA_PROTOCOL = f'{base_path}/ASVspoof2019_LA_cm_protocols/{protocol_file}'
        LA_DIR = f'{base_path}/ASVspoof2019_LA_{part}/flac/'

        dataset_df = pd.read_csv(LA_PROTOCOL, delimiter=' ', header=None).sample(frac=1, random_state=SEED).reset_index()
        audio_paths = [f'{LA_DIR}{path}.flac' for path in dataset_df[1]]
        dataset_labels = (dataset_df[4] != 'spoof').astype(int).to_numpy()

        return audio_paths, dataset_labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, ind):
        audio_path = self.audio_paths[ind]
        audio, sr = torchaudio.load(audio_path, format='flac')
        audio = audio[:, :self.max_len]
        length = audio.shape[1]
        label = self.dataset_labels[ind]
        return {
            "audio_path": audio_path,
            "label": label,
            "audio": audio,
            "sample_rate": sr,
            "length": length
        }

def collate_fn(dataset_items):
    audios = [elem["audio"][0] for elem in dataset_items]
    labels = [elem["label"] for elem in dataset_items]
    audios = pad_sequence(audios, batch_first=True).unsqueeze(1)
    return audios, torch.tensor(labels)
