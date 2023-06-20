"""
Contains the train_epoch and test_epoch functions
"""

import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import mean_squared_error as sklearn_mse

def train_epoch(model, dataloader, loss_fn, optimizer, epoch, train_slopes_true, year, target, device, t_cls=0.05, num_extra_features=0):
  """
  Trains the PyTorch model for one epoch. 
  Forward pass -> Optimizer zero grad -> Backpropagation -> Optimizer step

  Args:
  model: Pytorch model that will be trained.
  dataloader: PyTorch train dataloader that will be used to train the model.
  loss_fn: PyTorch loss function to be optimized by the optimizer.
  optimizer: PyTorch optimizer algorithm that will be used to optimize the loss function.
  epoch: Epoch that is currently being trained. Used for printing purposes.
  train_slopes_true: Not necessary as of yet. Will be used to compute ramp metric.
  year: Also not necessary yet, will be used to compute ramp metric.
  target: The label of each input, such as future_ghi, future_kt_solis and so on...
  t_cls: Also not necessary yet, will be used to compute ramp metric.
  device: Device that will do the computations (e.g. "cuda" or "cpu").
  num_extra_features: If the model takes the image plus some extra features as it's input, num_extra_features is how many extra features there are. 
  """

  model.train()
  train_loss, train_rmse = 0, 0
  with tqdm(dataloader, unit="batch") as tepoch:
    #X contains all the image data and aux contains all the scalar data, as it was when the PyTorch Dataset was created.
    for X, aux in tepoch: 
      #Printing purposes
      tepoch.set_description(f"Epoch{epoch}") 

      #time_stamps, ghi_cs, future_ghi_cs, future_ghi are all auxiliary tensors and are not used for training
      time_stamps = aux[:, 0].detach()
      ghi_cs = aux[:, 1].detach()
      future_ghi_cs = aux[:, 2].detach()
      future_ghi = aux[:, 3].detach()

      #y contains the label its first column and all the extra features in the subsequent columns
      y = aux[:, 4:].float()
      X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

      if num_extra_features > 0:
        y_pred = model(X, y[:, 1:])
      else:
        y_pred = model(X)

      #Calculate the loss
      loss = loss_fn(y_pred.squeeze(), y[:, 0])

      #Accumulate the loss, if the target is the future kt and not ghi, we need to rescale it back to ghi in order to print the loss curves 
      if "ghi" in target:
        train_loss += loss.item()
      else:
        train_loss += sklearn_mse(future_ghi, y_pred.squeeze().detach().to("cpu") * future_ghi_cs).item()
    
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      #Printing purposes
      tepoch.set_postfix(loss=train_loss)

  train_loss = train_loss / len(dataloader)
  train_rmse = np.sqrt(train_loss) #Loss = MSE
  return train_loss, train_rmse

def test_epoch(model, dataloader, loss_fn, test_slopes_true, year, target, device, t_cls=0.05, num_extra_features=0):
  """
  Test the PyTorch model for one epoch.
  """
