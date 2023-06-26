"""
Pytorch code to make all the models used in this project
Available models are:
-RegressionResNet18 - Base ResNet18 modified to fit a regression task.
"""
import torch
import torchvision
from torch import nn

class RegressionResNet18(nn.Module):
  """
  Base ResNet18 modified to fit a regression task. Expects an image as input, and will output a real number prediction.
  
  Args:
  weights: Initialize the weights of the model to be trained (torchvision.models.ResNet18_Weights.DEFAULT is recommended).
  dropout: Dropout hyperparameter indicating the probability of a unit to be shutdown during training (avoids overfitting).
  """
  def __init__(self, weights, dropout):
    super().__init__()
    self.resnet = torchvision.models.resnet18(weights=weights)
    for name, param in self.resnet.named_parameters():
      if 'bn' in name:
        param.requires_grad = False
        
      if dropout > 0:
        self.resnet.fc = nn.Sequential(
          nn.Dropout(p=dropout),
          nn.Linear(in_features=512,
                    out_features=1)
        )
      else:
        self.resnet.fc = nn.Linear(in_features=512,
                                   out_features=1)
        
  def forward(self, x):
    x = self.resnet(x)
    return x

class RegressionResNet50(nn.Module):
  """
  Base ResNet50 modified to fit a regression task. Expects an image as input, and will output a real number prediction.
  
  Args:
  weights: Initialize the weights of the model to be trained (torchvision.models.ResNet50_Weights.DEFAULT is recommended).
  dropout: Dropout hyperparameter indicating the probability of a unit to be shutdown during training (avoids overfitting).
  """
  def __init__(self, weights, dropout):
    super().__init__()
    self.resnet = torchvision.models.resnet50(weights=weights)
    for name, param in self.resnet.named_parameters():
      if 'bn' in name:
        param.requires_grad = False
        
      if dropout > 0:
        self.resnet.fc = nn.Sequential(
          nn.Dropout(p=dropout),
          nn.Linear(in_features=2048,
                    out_features=1)
        )
      else:
        self.resnet.fc = nn.Linear(in_features=2048,
                                   out_features=1)
        
  def forward(self, x):
    x = self.resnet(x)
    return x

class RegressionResNet101(nn.Module):
  """
  Base ResNet101 modified to fit a regression task. Expects an image as input, and will output a real number prediction.
  
  Args:
  weights: Initialize the weights of the model to be trained (torchvision.models.ResNet101_Weights.DEFAULT is recommended).
  dropout: Dropout hyperparameter indicating the probability of a unit to be shutdown during training (avoids overfitting).
  """
  def __init__(self, weights, dropout):
    super().__init__()
    self.resnet = torchvision.models.resnet101(weights=weights)
    for name, param in self.resnet.named_parameters():
      if 'bn' in name:
        param.requires_grad = False
        
      if dropout > 0:
        self.resnet.fc = nn.Sequential(
          nn.Dropout(p=dropout),
          nn.Linear(in_features=2048,
                    out_features=1)
        )
      else:
        self.resnet.fc = nn.Linear(in_features=2048,
                                   out_features=1)
        
  def forward(self, x):
    x = self.resnet(x)
    return x     