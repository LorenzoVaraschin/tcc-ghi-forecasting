"""
Pytorch code to create the Datasets used for training the models. Also contains the make_dataloaders function, that 
Datasets must be one of two:
1. ExtraFeaturesOnly - dataset to use when training a model without the images
2. ImagesAndExtraFeaturesOnly - dataset to use when training a model with the images
"""
import torch
import numpy as np
import pandas as pd
import cv2
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ExtraFeaturesOnlyDataset(Dataset):
  """
  Dataset that cointains only the extra features data and not the image data.

  Args:
  features: Input features to the model, at time t = datetime_index.
  label: Corresponding label to the input features, at time t = datetime_index + delta_t.
  datetime_index: Date time index of the features and labels, must be in pandas.DateTimeIndex format.
  ghi_cs: The clear sky ghi at t = datetime_index.
  """
  def __init__(self, features, label, datetime_index, ghi_cs):
    self.features = features
    self.label = label
    self.datetime_index = datetime_index
    self.ghi_cs = ghi_cs

  def __len__(self):
    return len(self.label)

  def __getitem__(self, index):
    features = self.features[index]
    label = self.label[index]
    time_stamp = self.datetime_index[index]
    time_stamp = torch.tensor(np.array([time_stamp.timestamp()]))
    current_ghi_cs = self.ghi_cs[index]
    return features, torch.cat((time_stamp, current_ghi_cs, label), dim=0)

class ImageAndExtraFeaturesDataset(Dataset):
  """
  Dataset that contains the image data and that may or may not contain extra feature data.

  Args:
  paths: Path where the image data is being stored.
  transform: Transform to be applied to the image.
  label: Corresponding label to the images, at time t = datetime_index + delta_t.
  datetime_index: Date time index of the features and labels, must be in pandas.DateTimeIndex format.
  ghi_cs: The clear sky ghi at t = datetime_index.
  future_ghi_cs: The clear sky ghi at t = datetime_index + delta_t.
  future_ghi: The ghi at t = datetime_index + delta_t. Used to plot loss curves in the ghi scale when target is kt. 
  extra_features: Extra features that are to be used when training the model. Defatult = None.
  """
  def __init__(self, paths, transform, label, datetime_index, ghi_cs, future_ghi_cs, future_ghi, sun_center=None, extra_features=None, rotation_angle=0):
    self.paths = paths
    self.transform = transform
    self.label = label
    self.extra_features = extra_features
    self.datetime_index = datetime_index 
    self.ghi_cs = ghi_cs 
    self.future_ghi_cs = future_ghi_cs
    self.future_ghi = future_ghi
    self.sun_center = sun_center

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    image_path = self.paths.iloc[index]
    time_stamp = self.datetime_index[index]
    current_ghi_cs = self.ghi_cs[index] 
    future_ghi_cs = self.future_ghi_cs[index]
    future_ghi = self.future_ghi[index]
    time_stamp = torch.tensor(np.array([time_stamp.timestamp()]))
    #When using stacked images to train model - 9 channels
    if len(image_path) == 3:
      img_0 = read_image(image_path[0])
      img_0 = img_0[:, 20:245, 10:235] #Crop image 
      img_0 = self.transform(img_0) #Other transforms

      img_1 = read_image(image_path[1])
      img_1 = img_1[:, 20:245, 10:235]
      img_1 = self.transform(img_1)

      img_2 = read_image(image_path[2])
      img_2 = img_2[:, 20:245, 10:235]
      img_2 = self.transform(img_2)
      img = torch.cat((img_0, img_1, img_2), dim=0)
    else: #Single images
      img = read_image(image_path)
      if rotation_angle != 0:
        img = torchvision.transforms.functional.rotate(img, rotation_angle)
      if self.sun_center != None:
        img = self.transform(img)
        mask = np.zeros((64, 64))
        mask = cv2.circle(mask, self.sun_center[index], 5, (255, 255, 255), -1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        img = torch.cat((img, mask), dim=0)
      else: 
        img = img[:, 20:245, 10:235]
        img = self.transform(img)
    ghi = self.label[index]
    if self.extra_features != None:
      extra_features = self.extra_features[index]
      return img, torch.cat((time_stamp, current_ghi_cs, future_ghi_cs, future_ghi, ghi, extra_features), dim=0)
    else:
      return img, torch.cat((time_stamp, current_ghi_cs, future_ghi_cs, future_ghi, ghi), dim=0)

class ImageSequenceDataset(Dataset):
  """
  Dataset that contains the sequence of images and the label for the sequence.

  Args:
  paths: Path where the image data is being stored.
  transform: Transform to be applied to the image.
  label: Corresponding label to the images, at time t = datetime_index + delta_t.
  datetime_index: Date time index of the features and labels, must be in pandas.DateTimeIndex format.
  ghi_cs: The clear sky ghi at t = datetime_index.
  future_ghi_cs: The clear sky ghi at t = datetime_index + delta_t.
  future_ghi: The ghi at t = datetime_index + delta_t. Used to plot loss curves in the ghi scale when target is kt. 
  extra_features: Extra features that are to be used when training the model. Defatult = None.
  """
  def __init__(self, paths, transform, label, datetime_index, ghi_cs, future_ghi_cs, future_ghi, extra_features=None):
    self.paths = paths
    self.transform = transform
    self.label = label
    self.datetime_index = datetime_index
    self.ghi_cs = ghi_cs
    self.future_ghi_cs = future_ghi_cs
    self.future_ghi = future_ghi
    self.extra_features = extra_features

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    image_paths = self.paths.iloc[index]
    time_stamp = self.datetime_index[index]
    current_ghi_cs = self.ghi_cs[index] 
    future_ghi_cs = self.future_ghi_cs[index]
    future_ghi = self.future_ghi[index]
    time_stamp = torch.tensor(np.array([time_stamp.timestamp()]))
    ghi = self.label[index]

    img_sequence = torch.tensor([])
    for path in image_paths:
      img = read_image(path)
      img = img[:, 20:245, 10:235] #Crop image 
      img = self.transform(img) #Other transforms
      img = img.unsqueeze(dim=0)
      img_sequence = torch.cat((img_sequence, img), dim=0)
      
    if self.extra_features != None:
      extra_features = self.extra_features[index]
      return img_sequence, torch.cat((time_stamp, current_ghi_cs, future_ghi_cs, future_ghi, ghi, extra_features), dim=0)
    else:
      return img_sequence, torch.cat((time_stamp, current_ghi_cs, future_ghi_cs, future_ghi, ghi), dim=0)

def make_dataloaders(
  df_train: pd.DataFrame,
  df_test: pd.DataFrame,
  df_val: pd.DataFrame,
  config: dict,
  sampler
):
  """
  Transforms a pandas DataFrame into a PyTorch DataLoader.
  These DataFrames should have the desired column names as presentend in columns.txt in the DataFrames folder.

  Args: 
  df_train: DataFrame used for training.
  df_test: DataFrame used for testing.
  df_val: DataFrame used for validation.
  config: Dictionary which contains all the hyperparameters used for training.
  shold have the same keys as presented in config.txt

  Returns:
  Three PyTorch DataLoader objects, in order train_dataloader, test_dataloader and val_dataloader
  """

  if config["rotate_image"] == True:
    img_transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"]), antialias=True), #Resize image
        transforms.RandomRotation((config["min_angle"], config["max_angle"])), #Rotate image
        transforms.ConvertImageDtype(dtype=torch.float32), #read_image() read each pixel into uint8, but all tensors must be in dtype=torch.float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], #ResNet normalization
                            std=[0.229, 0.224, 0.225])
    ])
  else:
    img_transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"]), antialias=True), #Resize image
        transforms.ConvertImageDtype(dtype=torch.float32), #read_image() read each pixel into uint8, but all tensors must be in dtype=torch.float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], #ResNet normalization
                            std=[0.229, 0.224, 0.225])
    ])
  
  if len(config["extra_features"]) > 0:  
    #List comprehension where each list contains len(config["extra_feautres"]) elements, and each element is a tensor of shape [n_samples, 1]
    train_extra_features_list = [torch.tensor(list(df_train[feature][0::config["sample_rate"]]), dtype=torch.float32).unsqueeze(1) for feature in config["extra_features"]] 
    test_extra_features_list = [torch.tensor(list(df_test[feature][0::config["sample_rate"]]), dtype=torch.float32).unsqueeze(1) for feature in config["extra_features"]]
    val_extra_features_list = [torch.tensor(list(df_val[feature][0::config["sample_rate"]]), dtype=torch.float32).unsqueeze(1) for feature in config["extra_features"]]

    #Concatenate the lists along dim=1, resulting in a tensor of shape [n_samples, len(config["extra_features"])]
    train_extra_features = torch.cat((train_extra_features_list), dim=1)
    test_extra_features = torch.cat((test_extra_features_list), dim=1)
    val_extra_features = torch.cat((val_extra_features_list), dim=1)
  else:
    train_extra_features = None
    test_extra_features = None
    val_extra_features = None

  if config["sun_mask"]:
    train_sun_center_list = list(df_train["sun_center"][0::config["sample_rate"]])
    test_sun_center_list = list(df_test["sun_center"][0::config["sample_rate"]])
    val_sun_center_list = list(df_val["sun_center"][0::config["sample_rate"]])
  else:
    train_sun_center_list = None
    test_sun_center_list = None
    val_sun_center_list = None
  
  if config["model_name"] != "RecursiveResNet18AndLSTM":
    train_data = ImageAndExtraFeaturesDataset(
      paths=df_train["path"][0::config["sample_rate"]] if not config["stacked"] else df_train[["path_t-2x", "path_t-x", "path_t"]],
      transform=img_transform,
      label=torch.tensor(list(df_train[config["target"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      datetime_index=df_train.index[0::config["sample_rate"]],
      ghi_cs=torch.tensor(list(df_train["current_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      future_ghi_cs=torch.tensor(list(df_train["future_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      future_ghi=torch.tensor(list(df_train["future_ghi"][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      sun_center=train_sun_center_list,
      extra_features=train_extra_features,
      rotation_angle=config["rotation_angle"]
    )

    test_data = ImageAndExtraFeaturesDataset(
      paths=df_test["path"][0::config["sample_rate"]] if not config["stacked"] else df_test[["path_t-2x", "path_t-x", "path_t"]],
      transform=img_transform,
      label=torch.tensor(list(df_test[config["target"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      datetime_index=df_test.index[0::config["sample_rate"]],
      ghi_cs=torch.tensor(list(df_test["current_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      future_ghi_cs=torch.tensor(list(df_test["future_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      future_ghi=torch.tensor(list(df_test["future_ghi"][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      sun_center=test_sun_center_list,
      extra_features=test_extra_features,
      rotation_angle=config["rotation_angle"]
    )

    val_data = ImageAndExtraFeaturesDataset(
      paths=df_val["path"][0::config["sample_rate"]] if not config["stacked"] else df_val[["path_t-2x", "path_t-x", "path_t"]],
      transform=img_transform,
      label=torch.tensor(list(df_val[config["target"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      datetime_index=df_val.index[0::config["sample_rate"]],
      ghi_cs=torch.tensor(list(df_val["current_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      future_ghi_cs=torch.tensor(list(df_val["future_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      future_ghi=torch.tensor(list(df_val["future_ghi"][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      sun_center=val_sun_center_list,
      extra_features=val_extra_features,
      rotation_angle=config["rotation_angle"]
    )
  else:
    train_data = ImageSequenceDataset(
      paths=df_train["path"][0::config["sample_rate"]] if not config["stacked"] else df_train[["path_t-2x", "path_t-x", "path_t"]],
      transform=img_transform,
      label=torch.tensor(list(df_train[config["target"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      datetime_index=df_train.index[0::config["sample_rate"]],
      ghi_cs=torch.tensor(list(df_train["current_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      future_ghi_cs=torch.tensor(list(df_train["future_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      future_ghi=torch.tensor(list(df_train["future_ghi"][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      extra_features=train_extra_features
    )

    test_data = ImageSequenceDataset(
      paths=df_test["path"][0::config["sample_rate"]] if not config["stacked"] else df_test[["path_t-2x", "path_t-x", "path_t"]],
      transform=img_transform,
      label=torch.tensor(list(df_test[config["target"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      datetime_index=df_test.index[0::config["sample_rate"]],
      ghi_cs=torch.tensor(list(df_test["current_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      future_ghi_cs=torch.tensor(list(df_test["future_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      future_ghi=torch.tensor(list(df_test["future_ghi"][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      extra_features=test_extra_features
    )

    val_data = ImageSequenceDataset(
      paths=df_val["path"][0::config["sample_rate"]] if not config["stacked"] else df_val[["path_t-2x", "path_t-x", "path_t"]],
      transform=img_transform,
      label=torch.tensor(list(df_val[config["target"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      datetime_index=df_val.index[0::config["sample_rate"]],
      ghi_cs=torch.tensor(list(df_val["current_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1),
      future_ghi_cs=torch.tensor(list(df_val["future_ghi_cs_"+config["cs_model"]][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      future_ghi=torch.tensor(list(df_val["future_ghi"][0::config["sample_rate"]]), dtype=torch.float64).unsqueeze(1), #used to plot loss curves in ghi scale
      extra_features=val_extra_features
    )

  train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    prefetch_factor=config["prefetch_factor"],
    pin_memory=config["pin_memory"],
    shuffle=True
  )

  test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    prefetch_factor=config["prefetch_factor"],
    pin_memory=config["pin_memory"],
    shuffle=False
  )

  val_dataloader = DataLoader(
    dataset=val_data,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    prefetch_factor=config["prefetch_factor"],
    pin_memory=config["pin_memory"],
    shuffle=False
  )

  train_features, train_labels = next(iter(train_dataloader))
  print(f"Feature batch shape: {train_features.size()}")
  print(f"Labels batch shape: {train_labels.size()}")

  return train_dataloader, test_dataloader, val_dataloader
