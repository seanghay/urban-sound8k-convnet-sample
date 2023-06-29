import torch
import os
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

class UrbanSoundDataset(Dataset):
  
  def __init__(self, manifest_file, audio_dir, transform, target_sr, num_samples, device):
    super().__init__()
    
    self.device = device
    self.annotations = pd.read_csv(manifest_file)
    self.audio_dir = audio_dir
    self.transform = transform.to(self.device)
    self.target_sr = target_sr
    self.num_samples = num_samples
    self.class_mappings = self.annotations[['classID', 'class']].groupby("classID").first().reset_index(drop=True)
    self.class_mappings = self.class_mappings.to_dict()['class']  
    
  def _resample(self, signal, sr):    
    if sr != self.target_sr:
      resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
      resampler = resampler.to(self.device)
      return resampler(signal)
    return signal  
    
  def _mix_down_mono(self, signal):
    if signal.shape[0] > 1:
      signal = torch.mean(signal, dim=0, keepdim=True)
    return signal
  
  def __len__(self):
    return len(self.annotations)
  
  def __getitem__(self, index):
    label = self._audio_sample_label(index)
    signal, sr = torchaudio.load(self._audio_sample_path(index))
    # move to cpu or cuda
    signal = signal.to(self.device)
    # resample to target sampling rate
    signal = self._resample(signal, sr)  
    # down mix to mono
    signal = self._mix_down_mono(signal)
    # cut
    signal = self._cut(signal)
    # padding
    signal = self._pad(signal)
    # apply transformation
    signal = self.transform(signal)
    return signal, label
  
  def _pad(self, signal):
    if signal.shape[1] < self.num_samples:
      last_dim_padding = (0, self.num_samples - signal.shape[1])
      return torch.nn.functional.pad(signal, last_dim_padding)
    return signal  

  def _cut(self, signal):
    if signal.shape[1] > self.num_samples:
      return signal[:, :self.num_samples]
    return signal
    
  def _audio_sample_path(self, index):
    file_name = self.annotations.iloc[index, 0]    
    num_fold = self.annotations.iloc[index, 5]
    return os.path.join(self.audio_dir, f"fold{num_fold}", file_name)
        
  def _audio_sample_label(self, index):
    return self.annotations.iloc[index, 6]

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  target_sr = 22050
  num_samples = 22050
  
  mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=target_sr,
    n_fft=1024,
    hop_length=512,
    n_mels=64
  )
  
  ds = UrbanSoundDataset(
    manifest_file="data/UrbanSound8K.csv", 
    audio_dir="data/",
    transform=mel_transform,
    target_sr=target_sr,
    num_samples=num_samples,
    device=device
  )
  
  len(ds)
  signal, label = ds[0]  
  print(signal.shape, label)