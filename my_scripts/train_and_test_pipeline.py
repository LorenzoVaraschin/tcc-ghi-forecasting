"""
Contains the train_epoch and test_epoch functions
"""

import torch
import numpy as np
import wandb
import pandas as pd 

from tqdm import tqdm
from sklearn.metrics import mean_squared_error as sklearn_mse
from sklearn.metrics import mean_absolute_error as sklearn_mae

def train_epoch(model, dataloader, loss_fn, optimizer, epoch, target, device, t_cls=0.05, num_extra_features=0):
  """
  Trains the PyTorch model for one epoch. 
  Forward pass -> Optimizer zero grad -> Backpropagation -> Optimizer step

  Args:
  model: Pytorch model that will be trained.
  dataloader: PyTorch train dataloader that will be used to train the model.
  loss_fn: PyTorch loss function to be optimized by the optimizer.
  optimizer: PyTorch optimizer algorithm that will be used to optimize the loss function.
  epoch: Epoch that is currently being trained. Used for printing purposes.
  target: The label of each input, such as future_ghi, future_kt_solis and so on...
  t_cls: Also not necessary yet, will be used to compute ramp metric.
  device: Device that will do the computations (e.g. "cuda" or "cpu").
  num_extra_features: If the model takes the image plus some extra features as it's input, num_extra_features is how many extra features there are.

  Returns:
  train_loss and train_rmse 
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

      #Forward pass
      if num_extra_features > 0:
        y_pred = model(X, y[:, 1:])
      else:
        y_pred = model(X)

      #Calculate the loss
      loss = loss_fn(y_pred.squeeze(), y[:, 0])

      #Accumulate the loss, if the target is the future kt and not ghi rescale it back to ghi in order to print the loss curves 
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

def test_epoch(model, dataloader, loss_fn, target, device, dataset="test", t_cls=0.05, num_extra_features=0):
  """
  Test the PyTorch model for one epoch.

  Args:
  model: Pytorch model that will be evaluated.
  dataloader: PyTorch test dataloader that will be used to evaluate the model.
  loss_fn: Same PyTorch loss function that was used during training.
  test_slopes_true: Not necessary as of yet. Will be used to compute ramp metric.
  year: Also not necessary yet, will be used to compute ramp metric.
  target: The label of each input, such as future_ghi, future_kt_solis and so on...
  dataset: Validation or test dataset. Default: test.
  t_cls: Also not necessary yet, will be used to compute ramp metric.
  device: Device that will do the computations (e.g. "cuda" or "cpu").
  num_extra_features: If the model takes the image plus some extra features as it's input, num_extra_features is how many extra features there are.

  Returns:
  test_loss, test_rmse, test_mae, predictions_ghi, predictions_kt and pred_timestamps for the test dataset. val_loss, val_rmse for the val dataset. 
  """
  model.eval()
  test_loss, test_rmse, test_mae = 0, 0, 0
  predictions_ghi, predictions_kt = torch.tensor([]), torch.tensor([])
  pred_timestamps = pd.DatetimeIndex([])
  with torch.inference_mode():
    #X contains all the image data and aux contains all the scalar data, as it was when the PyTorch Dataset was created.
    for X, aux in dataloader:

      #time_stamps, ghi_cs, future_ghi_cs, future_ghi are all auxiliary tensors and are not used for training
      time_stamps = aux[:, 0].detach()
      ghi_cs = aux[:, 1].detach()
      future_ghi_cs = aux[:, 2].detach()
      future_ghi = aux[:, 3].detach()

      #y contains the label its first column and all the extra features in the subsequent columns
      y = aux[:, 4:].float()
      X, y = X.to(device), y.to(device)

      #Forward pass
      if num_extra_features > 0:
        test_pred = model(X, y[:, 1:])
      else:
        test_pred = model(X)

      #Calculate the loss
      loss = loss_fn(test_pred.squeeze(), y[:, 0])

      #Accumulate the loss, if the target is the future kt and not ghi rescale it back to ghi in order to print the loss curves
      if "ghi" in target:
        test_loss += loss.item()
        if dataset == "test":
          test_mae += sklearn_mae(y[:, 0].detach().to("cpu"), test_pred.squeeze().detach().to("cpu")).item()
          predictions_ghi = torch.cat((predictions_ghi, test_pred.squeeze().detach().to("cpu")), dim=0)
          pred_timestamps = pred_timestamps.append(pd.to_datetime(time_stamps.tolist(), unit='s'))
      else:
        test_loss += sklearn_mse(future_ghi, test_pred.squeeze().detach().to("cpu") * future_ghi_cs).item()
        if dataset == "test":
          test_mae += sklearn_mae(future_ghi, test_pred.squeeze().detach().to("cpu") * future_ghi_cs).item()
          predictions_ghi = torch.cat((predictions_ghi, test_pred.squeeze().detach().to("cpu") * future_ghi_cs), dim=0)
          predictions_kt = torch.cat((predictions_kt, test_pred.squeeze().detach().to("cpu")), dim=0)
          pred_timestamps = pred_timestamps.append(pd.to_datetime(time_stamps.tolist(), unit='s'))

  test_loss = test_loss / len(dataloader)
  test_rmse = np.sqrt(test_loss)

  if dataset == "test":
    test_mae = test_mae/len(dataloader)
    return test_loss, test_rmse, test_mae, predictions_ghi, predictions_kt, pred_timestamps
  else:
    return test_loss, test_rmse

def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, scheduler_name, epochs, t_cls, target, device, VAL_LOSS, MODEL_CHECKPOINT=False, num_extra_features=0):
  """
  Trains and validates the model for the number of epochs specified.

  Args:
  model: Pytorch model that will be trained.
  train_dataloader: PyTorch train dataloader that will be used to train the model.
  val_dataloader: PyTorch validation dataloader that will be used to validate the model.
  optimizer: PyTorch optimizer algorithm that will be used to optimize the loss function.
  loss_fn: PyTorch loss function to be optimized by the optimizer.
  scheduler: PyTorch learning rate scheduler.
  scheduler_name: Name of the PyTorch learning rate scheduler.
  epochs: Number of epochs to train the model.
  t_cls: Not necessary yet, will be used to compute ramp metric.
  target: The label of each input, such as future_ghi, future_kt_solis and so on...
  device: Device that will do the computations (e.g. "cuda" or "cpu").
  VAL_LOSS: Validation loss to check for model checkpoint. Default=0 (won't save model checkpoint).
  MODEL_CHECKPOINT: Whether or not model's checkpoint will be saved
  num_extra_features: If the model takes the image plus some extra features as it's input, num_extra_features is how many extra features there are.
  """
  wandb.watch(model, loss_fn, log="all", log_freq=100)
  PATH = "/content/model_checkpoint.pt" #Path to sabe model checkpoint
  for epoch in range(epochs):
    if (epoch > 0) & (MODEL_CHECKPOINT == True):
      #Load the saved checkopoint
      checkpoint = torch.load(PATH)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      VAL_LOSS = checkpoint['val_loss']

    train_loss, train_rmse = train_epoch(
      model=model,
      dataloader=train_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      epoch=epoch,
      target=target,
      device=device,
      t_cls=t_cls,
      num_extra_features=num_extra_features
    )

    val_loss, val_rmse = test_epoch(
      model=model,
      dataloader=val_dataloader,
      loss_fn=loss_fn,
      target=target,
      device=device,
      dataset="val",
      t_cls=t_cls,
      num_extra_features=num_extra_features
    )
    wandb.log({"train_rmse": train_rmse, "val_rmse": val_rmse, "lr": optimizer.param_groups[0]['lr']})
    print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f} | Val loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | LR: {optimizer.param_groups[0]['lr']}")
    if "reduce_lr_on_plateau" in scheduler_name:
      scheduler.step(val_loss)
    else:
      scheduler.step()

    if val_loss < VAL_LOSS:
      #Save the new checkpoint
      torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
            }, PATH)