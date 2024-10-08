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
  def __init__(self, weights, dropout, stacked=False, sun_mask=False):
    super().__init__()
    self.resnet = torchvision.models.resnet18(weights=weights)
    if stacked:
      self.resnet.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if sun_mask:
      self.resnet.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

class RegressionUNet(nn.Module):
  def __init__(self, num_filters=12, dropout=0.4, image_size=(64, 64)):
    super().__init__()
    self.top_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=(1, 1), padding='same'),
        nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(num_filters),
        nn.ReLU()
    )
    self.mid_layer1 = nn.Sequential(
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        nn.Conv2d(in_channels=num_filters, out_channels=2*num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(2*num_filters),
        nn.ReLU()
    )
    self.bot_layer1 = nn.Sequential(
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        nn.Conv2d(in_channels=2*num_filters, out_channels=4*num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(4*num_filters),
        nn.ReLU()
    )
    self.bottleneck1 = nn.Sequential(
        nn.Conv2d(in_channels=4*num_filters, out_channels=4*num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(4*num_filters),
        nn.ReLU(),
        nn.Conv2d(in_channels=4*num_filters, out_channels=4*num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(4*num_filters)
    )
    self.bottlenck2 = nn.Sequential(
        nn.Conv2d(in_channels=4*num_filters, out_channels=4*num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(4*num_filters),
        nn.ReLU(),
        nn.Conv2d(in_channels=4*num_filters, out_channels=4*num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(4*num_filters)
    )
    self.bot_upsample = nn.Sequential(
        nn.Upsample(scale_factor=(2, 2), mode='nearest'),
        nn.Conv2d(in_channels=4*num_filters, out_channels=2*num_filters, kernel_size=(3, 3), padding='same')
    )
    self.mid_layer2 = nn.Sequential(
        nn.Conv2d(in_channels=4*num_filters, out_channels=2*num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(2*num_filters),
        nn.ReLU(),
        nn.Dropout(p=dropout)
    )
    self.mid_upsample = nn.Sequential(
        nn.Upsample(scale_factor=(2, 2), mode='nearest'),
        nn.Conv2d(in_channels=2*num_filters, out_channels=num_filters, kernel_size=(3, 3), padding='same')
    )
    self.top_layer2 = nn.Sequential(
        nn.Conv2d(in_channels=2*num_filters, out_channels=num_filters, kernel_size=(3, 3), padding='same'),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(),
        nn.Dropout(p=dropout)
    )
    self.flatten_layer = nn.Sequential(
        nn.Conv2d(in_channels=num_filters, out_channels=1, kernel_size=(1, 1), padding='same'),
        nn.ReLU(),
        nn.Flatten()
    )
    self.linear_regressor = nn.Linear(in_features=image_size[0]*image_size[0], out_features=1)
  def forward(self, x):
    top_output1 = self.top_layer1(x)

    mid_output1 = self.mid_layer1(top_output1)

    bot_output1 = self.bot_layer1(mid_output1)

    identity = bot_output1
    bottleneck_out = self.bottleneck1(bot_output1)
    bottleneck_out += identity

    identity = bottleneck_out
    bottleneck_out = self.bottlenck2(bottleneck_out)
    bottleneck_out += identity
    
    mid_output2 = self.bot_upsample(bottleneck_out)
    mid_output2 = self.mid_layer2(torch.cat((mid_output2, mid_output1), axis=1))

    top_output2 = self.mid_upsample(mid_output2)
    top_output2 = self.top_layer2(torch.cat((top_output2, top_output1), axis=1))
    top_output2 = self.flatten_layer(top_output2)

    prediction = self.linear_regressor(top_output2)
    return prediction
    
class RegressionViT32(nn.Module):
  def __init__(self, image_size=224, weights="ViT_B_32_Weights.DEFAULT"):
    super().__init__()
    self.vit = torchvision.models.vit_b_32(weights=weights, image_size=image_size)  
    self.vit.heads = nn.Sequential(
      nn.Linear(in_features=768, out_features=1)
    )
        
  def forward(self, x):
    x = self.vit(x)
    return x


class RecursiveResNet18AndLSTM(nn.Module):
  """
  Base ResNet18 modified to fit a regression task. Expects an image as input, and will output a real number prediction.
  
  Args:
  weights: Initialize the weights of the model to be trained (torchvision.models.ResNet18_Weights.DEFAULT is recommended).
  dropout: Dropout hyperparameter indicating the probability of a unit to be shutdown during training (avoids overfitting).
  stacked: Wether or not the input images are stacked images (a stacked image is an input composed of 3 stacked images, resulting in a input with 9 channels).
  """
  def __init__(self, weights, dropout, hidden_dim, lstm_layers, device):
    super().__init__()
    self.device = device
    self.hidden_dim = hidden_dim
    self.resnet = torchvision.models.resnet18(weights=weights)
    for name, param in self.resnet.named_parameters():
      if 'bn' in name:
        param.requires_grad = False

    self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    
    self.lstm = nn.LSTM(
        input_size=512,
        hidden_size=hidden_dim,
        batch_first=False,
        dropout=dropout,
        num_layers=lstm_layers
    )

    self.predictor = nn.Linear(
        in_features=hidden_dim,
        out_features=1
    )

  def forward(self, x):
    #x has shape (batch_size, seq_size, n_channels, image_height, image_width)
    resnet_embeddings_seq = torch.tensor([]).to(self.device)
    for image_batch in x.view(x.size()[1], x.size()[0], x.size()[2], x.size()[3], x.size()[4]):
      embedding_batch = self.resnet(image_batch).squeeze()
      resnet_embeddings_seq = torch.cat((resnet_embeddings_seq, embedding_batch.unsqueeze(dim=0)), dim=0)
    h0, c0 = torch.randn(1, resnet_embeddings_seq.size(1), self.hidden_dim).to(self.device), torch.randn(1, resnet_embeddings_seq.size(1), self.hidden_dim).to(self.device)
    output, (hn, cn) = self.lstm(resnet_embeddings_seq, (h0, c0))
    output = output[-1, :, :]
    prediction = self.predictor(output)
    return prediction

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
