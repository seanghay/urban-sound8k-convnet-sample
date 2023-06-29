import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import UrbanSoundDataset
from cnn import ConvNet
import torchaudio

def train_epoch(model, data_loader, loss_fn, optimizer, device):
  
  for inputs, targets in data_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    preds = model(inputs)
    loss = loss_fn(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epoch):
  
  for i in range(epoch):
    print(f"Epoch: {i + 1}")
    train_epoch(model, data_loader, loss_fn, optimizer, device)
    print("-----")
    
  print("Training done")

if __name__ == "__main__":
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  num_epoch = 100
  batch_size = 2048
  target_sr = 22050
  num_samples = 22050
  loss_fn = nn.CrossEntropyLoss()
  
  mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=target_sr,
    n_fft=1024,
    hop_length=512,
    n_mels=64
  )
  
  train_data = UrbanSoundDataset(
    manifest_file="data/UrbanSound8K.csv", 
    audio_dir="data/",
    transform=mel_transform,
    target_sr=target_sr,
    num_samples=num_samples,
    device=device
  )
  
  train_data_loader = DataLoader(train_data, batch_size=batch_size)
  print(f"Device: {device}")

  convnet = ConvNet().to(device)
  optimizer = torch.optim.Adam(convnet.parameters(), lr=0.001)

  train(convnet, train_data_loader, device=device, epoch=num_epoch, optimizer=optimizer, loss_fn=loss_fn)
  torch.save(convnet.state_dict(), "./convnet.pth")
  print("Model saved.")