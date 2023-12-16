import torch
import torchaudio

print(torch.cuda.is_available())
print(torchaudio.list_audio_backends())