"""
Creates the model's training and testing pipeline
"""
import torch
import wandb

from make_dataloader import make_dataloaders
from make_models import RegressionResNet18, RecursiveResNet18AndLSTM, RegressionResNet18EmbedTransform, RegressionResNet50, RegressionResNet18ExtraFeatures, SunsetModel, RegressionVGG16
from train_and_test_pipeline import train_model, test_epoch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchinfo import summary


def model_pipeline(project, run_name, hyperparameters, df_train, df_test, df_val):
  """
  Creates the model and dataloaders and trains/tests the model. Log the results in wandb.

  Args:
  Project: Name of the project to be logged in wandb. 
  hyperparameters: Hyperparamaters of the model and config of the wandb log.
  df_train: Train Pandas DataFrame that contains the data for the train set.
  df_test: Test Pandas DataFrame that contains the data for the test set.
  df_val: Validation Panadas DataFrame that contains the data for the validation set.

  Returns:
  Trained model.
  """

  with wandb.init(project=project, name=run_name, config=hyperparameters):
    config = wandb.config

    #Create the dataloaders
    train_dataloader, test_dataloader, val_dataloader = make_dataloaders(df_train=df_train, df_test=df_test, df_val=df_val, config=hyperparameters)
#weights, dropout, hidden_dim, lstm_layers, device
    #Create the model
    assert hyperparameters["model_name"] in ["RegressionResNet18", "RegressionResNet18EmbedTransform", "RegressionResNet50", "RegressionResNet18ExtraFeatures", "SunsetModel", "RegressionVGG16"], "Invalid model name"
    if hyperparameters["model_name"] == "RegressionResNet18":
      model = RegressionResNet18(weights=hyperparameters["weights"], dropout=hyperparameters["dropout"], stacked=hyperparameters["stacked"], sun_mask=hyperparameters["sun_mask"]).to(hyperparameters["device"])
    elif hyperparameters["model_name"] == "RecursiveResNet18AndLSTM":
      model = RecursiveResNet18AndLSTM(weights=hyperparameters["weights"], dropout=hyperparameters["dropout"], hidden_dim=hyperparameters["hidden_dim"], lstm_layers=hyperparameters["lstm_layers"], device=hyperparameters["device"]).to(hyperparameters["device"])
    elif hyperparameters["model_name"] == "RegressionResNet18EmbedTransform":
      model = RegressionResNet18EmbedTransform(weights=hyperparameters["weights"], dropout=hyperparameters["dropout"], hidden_units=hyperparameters["hidden_units"], fine_tuning=hyperparameters["fine_tuning"], stacked=hyperparameters["stacked"]).to(hyperparameters["device"])  
    elif hyperparameters["model_name"] == "RegressionResNet50":
      model = RegressionResNet50(weights=hyperparameters["weights"], dropout=hyperparameters["dropout"], stacked=hyperparameters["stacked"]).to(hyperparameters["device"])
    elif hyperparameters["model_name"] == "RegressionResNet18ExtraFeatures":
      model = RegressionResNet18ExtraFeatures(num_extra_features=hyperparameters["num_extra_features"], weights=hyperparameters["weights"],
                                              dropout=hyperparameters["dropout"], stacked=hyperparameters["stacked"]).to(hyperparameters["device"])
    elif hyperparameters["model_name"] == "SunsetModel":
      model = SunsetModel(dropout=hyperparameters["dropout"])
    elif hyperparameters["model_name"] == "RegressionVGG16":
      model = RegressionVGG16(weights=hyperparameters["weights"], dropout=hyperparameters["dropout"])

    model_sum = summary(model=model, input_size=hyperparameters["input_shape"], col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20, row_settings=["var_names"])
    print(model_sum)
    
    #Create the loss function
    assert hyperparameters["loss_fn"] in ["mse_loss", "mae_loss", "huber_loss"], "Invalid loss function"
    if hyperparameters["loss_fn"] == "mse_loss":
      loss_fn = torch.nn.MSELoss()
    elif hyperparameters["loss_fn"] == "mae_loss":
      loss_fn = torch.nn.L1Loss()
    elif hyperparameters["loss_fn"] == "huber_loss":
      loss_fn = torch.nn.HuberLoss(delta=hyperparameters["huber_delta"])

    assert hyperparameters["optimizer"] in ["adam"], "Invalid optimizer"
    if hyperparameters["optimizer"] == "adam":
      optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=hyperparameters["learning_rate"]
      )

    #Create the scheduler
    assert hyperparameters["scheduler"] in ["step_lr", "reduce_lr_on_plateau"], "Invalid Scheduler"
    if hyperparameters["scheduler"] == "step_lr":
      scheduler = StepLR(optimizer, step_size=hyperparameters["scheduler_step_size"], gamma=hyperparameters["scheduler_factor"])
    elif hyperparameters["scheduler"] == "reduce_lr_on_plateau":
      scheduler = ReduceLROnPlateau(optimizer, 'min', factor=hyperparameters["scheduler_factor"], patience=hyperparameters["scheduler_patience"])

    #Create the model checkpoint
    if hyperparameters["model_checkpoint"] == True:
      VAL_LOSS = 1e10
      print("\nModel checkpoint is on, epoch with best validation loss will be saved to disk on /content/model_checkpoint.pt\n")
    else :
      VAL_LOSS = 0 #Will never update model's checkpoint
      
    #Train the model
    train_model(
      model=model,
      train_dataloader=train_dataloader,
      val_dataloader=val_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      scheduler=scheduler,
      scheduler_name=hyperparameters["scheduler"],
      epochs=hyperparameters["epochs"],
      t_cls=hyperparameters["t_cls"],
      target=hyperparameters["target"],
      device=hyperparameters["device"],
      VAL_LOSS=VAL_LOSS,
      MODEL_CHECKPOINT=hyperparameters["model_checkpoint"],
      num_extra_features=hyperparameters["num_extra_features"]   
    )

    #Test the model
    test_loss, test_rmse, test_mae, predictions_ghi, predictions_kt, pred_timestamps = test_epoch(
      model=model,
      dataloader=test_dataloader,
      loss_fn=loss_fn,
      target=hyperparameters["target"],
      device=hyperparameters["device"],
      dataset="test",
      t_cls=hyperparameters["t_cls"],
      num_extra_features=hyperparameters["num_extra_features"]
    )

    wandb.log({"test_rmse": test_rmse, "test_mae": test_mae})

    # Save the model in the exchangeable ONNX format
    train_features, train_labels = next(iter(train_dataloader))
    if hyperparameters["num_extra_features"] > 0:
      args = (train_features.to(hyperparameters["device"]), {"extra_features":train_labels[:, 5:].float().to(hyperparameters["device"])})
    else:
      args = (train_features.to(hyperparameters["device"]))
    torch.onnx.export(model, args, "model.onnx")
    wandb.save("model.onnx")

    return model, predictions_ghi, predictions_kt, pred_timestamps
