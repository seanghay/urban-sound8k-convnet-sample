import torch
from torch import nn
from torchsummary import summary

class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    # Layer 1
    self.conv1 = nn.Sequential(
      nn.Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=2
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    
    # Layer 2
    self.conv2 = nn.Sequential(
      nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=2
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    
    # Layer 3
    self.conv3 = nn.Sequential(
      nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=2
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    
    # Layer 4
    self.conv4 = nn.Sequential(
      nn.Conv2d(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=2
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    
    self.flatten = nn.Flatten()
    # 10 = classes, 128 = output from the final conv4
    # MaxPool2d-12 [-1, 128, 5, 4] 0  
    self.linear = nn.Linear(128 * 5 * 4, 10)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input):
    x = self.conv1(input)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flatten(x)
    logits = self.linear(x)
    predictions = self.softmax(logits)
    return predictions
  
if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  convnet = ConvNet().to(device)
  summary(convnet, (1, 64, 44), device=device)