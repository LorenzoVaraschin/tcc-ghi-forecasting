"""
Pytorch code to make all the models used in this project
Available models are:
-RegressionResNet18 - Base ResNet18 modified to fit a regression task.
-RegresionResNet50 - Base ResNet50 modified to fit a regression task.
-RegressionResNet18ExtraFeatures - ResNet18 modified to accept extra input features.
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
  stacked: Wether or not the input images are stacked images (a stacked image is an input composed of 3 stacked images, resulting in a input with 9 channels).
  """
  def __init__(self, weights, dropout, stacked=False):
    super().__init__()
    self.resnet = torchvision.models.resnet18(weights=weights)
    if stacked:
      self.resnet.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
  stacked: Wether or not the input images are stacked images (a stacked image is an input composed of 3 stacked images, resulting in a input with 9 channels).
  """
  def __init__(self, weights, dropout, stacked=False):
    super().__init__()
    self.resnet = torchvision.models.resnet50(weights=weights)
    if stacked:
      self.resnet.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

class RegressionResNet18ExtraFeatures(nn.Module):
  """
  ResNet18 modified to fit regresion task and to accomodate extra features in training. Expects an image and at least one extra feature as input.

  Args:
  num_extra_features: How many extra features will be used in training.
  weights: Initialize the weights of the model to be trained (torchvision.models.ResNet18_Weights.DEFAULT is recommended).
  dropout: Dropout hyperparameter indicating the probability of a unit to be shutdown during training (avoids overfitting).
  stacked: Wether or not the input images are stacked images (a stacked image is an input composed of 3 stacked images, resulting in a input with 9 channels).
  """
  def __init__(self, num_extra_features, dropout, weights, stacked=False):
    super().__init__()
    self.resnet = torchvision.models.resnet18(weights=weights)
    if stacked:
      self.resnet.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.resnet.fc = nn.Flatten()
    for name, param in self.resnet.named_parameters():
      if 'bn' in name:
        param.requires_grad = False

    self.new_layers_1 = nn.Sequential(
        nn.Linear(in_features=512,
                  out_features=64),
        nn.ReLU(),
    )
    self.new_layers_2 = nn.Sequential(
        nn.Linear(in_features=num_extra_features,
                  out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16,
                  out_features=16),
        nn.ReLU()
    )
    
    if dropout > 0:
      self.new_layers_3 = nn.Sequential(
        nn.Linear(in_features=80,
                  out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64,
                  out_features=32),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=32,
                  out_features=1)
    )
    else:
      self.new_layers_3 = nn.Sequential(
        nn.Linear(in_features=80,
                  out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64,
                  out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32,
                  out_features=1)
    )

  def forward(self, x, extra_features):
    x = self.resnet(x)
    x = self.new_layers_1(x)
    extra_features = self.new_layers_2(extra_features)
    x = torch.cat((x, extra_features), dim=1)
    x = self.new_layers_3(x)
    return x
