import torchaudio
import torch
from dataset import UrbanSoundDataset
from cnn import ConvNet

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = ConvNet().to(device)
  
  state_dict = torch.load("./convnet.pth")
  model.load_state_dict(state_dict)
  
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
  
  # first item
  
  for idx in range(100):  
    input, target_idx = ds[idx][0], ds[idx][1]
    
    # What's this?
    input.unsqueeze_(0)
    
    model.eval()
    
    with torch.no_grad():
      preds = model(input)
      predicted_idx = preds[0].argmax(0).to("cpu").item()
      print(ds.class_mappings[predicted_idx], ds.class_mappings[target_idx])
      print(ds.class_mappings[predicted_idx] == ds.class_mappings[target_idx])