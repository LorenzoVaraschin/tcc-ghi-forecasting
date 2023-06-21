"""
Creates the model's training and testing pipeline
"""
import torch
import wandb

from make_dataloader import make_dataloaders
from make_models import RegressionResNet18
from train_and_test_pipeline import train_model, test_epoch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def model_pipeline(project, hyperparameters, df_train, df_test, df_val):
  """
  Creates the model and dataloaders and trains/tests the model. Log the results in wandb.

  Args:
  Project: Name of the project to be logged in wandb. 
  hyperparameters: Hyperparamaters of the model and config of the wandb log.

  Returns:
  Trained model.
  """

  with wandb.init(project=project, config=hyperparameters):
    config = wandb.config

    #Create the dataloaders
    train_dataloader, test_dataloader, val_dataloader = make_dataloaders(df_train=df_train, df_test=df_test, df_val=df_val, config=hyperparameters)

    #Create the model
    model = RegressionResNet18(weights=hyperparameters["weights"], dropout=hyperparameters["dropout"]).to(hyperparameters["device"])

    #Create the loss function
    assert hyperparameters["loss_fn"] in ["mse_loss"], "Invalid loss function"
    if hyperparameters["loss_fn"] == "mse_loss":
      loss_fn = torch.nn.MSELoss()

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
      num_extra_features=hyperparameters["num_extra_features"]
    )

    #Test the model
    test_loss, test_rmse, test_mae = test_epoch(
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
    torch.onnx.export(model, train_features, "model.onnx")
    wandb.save("model.onnx")

    return model
    

