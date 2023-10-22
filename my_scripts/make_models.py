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


class RegressionResNet18EmbedTransform(nn.Module):
  """
  Base ResNet18 modified to fit a regression task and to change 512 dimension embedding. Expects an image as input, and will output a real number prediction.
  
  Args:
  weights: Initialize the weights of the model to be trained (torchvision.models.ResNet18_Weights.DEFAULT is recommended).
  dropout: Dropout hyperparameter indicating the probability of a unit to be shutdown during training (avoids overfitting).
  hidden_units: Number of hidden units to change the embedding size to.
  stacked: Wether or not the input images are stacked images (a stacked image is an input composed of 3 stacked images, resulting in a input with 9 channels).
  """
  def __init__(self, weights, dropout, hidden_units, fine_tuning, stacked=False):
    super().__init__()
    self.resnet = torchvision.models.resnet18(weights=weights)
    if stacked:
      self.resnet.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)

    frozen_layers = 0
    for name, param in self.resnet.named_parameters():
      if frozen_layers < fine_tuning:
        param.requires_grad = False
        frozen_layers += 1
      elif 'bn' in name:
        param.requires_grad = False
        
      if dropout > 0:
        self.resnet.fc = nn.Sequential(
          nn.Linear(in_features=512,
                    out_features=hidden_units),
          nn.ReLU()
        )
        self.linear = nn.Sequential(
          nn.Dropout(p=dropout),
          nn.Linear(in_features=hidden_units,
                    out_features=1)
        )
      else:
        self.resnet.fc = nn.Sequential(
          nn.Linear(in_features=512,
                    out_features=hidden_units),
          nn.ReLU()
        )
        self.linear = nn.Linear(in_features=hidden_units,
                                    out_features=1)
  def forward(self, x):
    x = self.resnet(x)
    x = self.linear(x)
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

class RegressionVGG16(nn.Module):
  """
  Base VGG16 modified to fit a regression task. Expects an image as input, and will output a real number prediction.
  
  Args:
  weights: Initialize the weights of the model to be trained (torchvision.models.VGG16_Weights.DEFAULT is recommended).
  dropout: Dropout hyperparameter indicating the probability of a unit to be shutdown during training (avoids overfitting).
  stacked: Wether or not the input images are stacked images (a stacked image is an input composed of 3 stacked images, resulting in a input with 9 channels).
  """
  def __init__(self, weights, dropout, stacked=False):
    super().__init__()
    self.vgg = torchvision.models.vgg16(weights=weights)

    if dropout > 0:
      self.vgg.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features=25088,
                  out_features=1)
      )
    else:
      self.vgg.classifier = nn.Linear(in_features=25088,
                                  out_features=1)
        
  def forward(self, x):
    x = self.vgg(x)
    return x

class SunsetModel(nn.Module):
  """
  Sunset model from `SKIPP’D: A SKy Images and Photovoltaic Power Generation Dataset for short-term solar forecasting` <https://doi.org/10.1016/j.solener.2023.03.043>

  Args:
  dropout: Dropout hyperparameter indicating the probability of a unit to be shutdown during training.
  """
  def __init__(self, dropout):

    super().__init__()

    self.conv_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=3,
                  out_channels=24,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=24),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.conv_layer2 = nn.Sequential(
        nn.Conv2d(in_channels=24,
                  out_channels=48,
                  kernel_size=3,
                  padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=48),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.dense_layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=48*16*16,
                  out_features=1024),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=1024,
                  out_features=1024),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=1024,
                  out_features=1)
    )

  def forward(self, x):
    x = self.conv_layer1(x)
    x = self.conv_layer2(x)
    x = self.dense_layer(x)
    return x
