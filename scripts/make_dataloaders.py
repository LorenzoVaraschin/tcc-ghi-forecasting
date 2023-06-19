"""
Pytorch code to create the Datasets and Dataloaders used for training the models
Datasets must be one of two:
1. ExtraFeaturesOnly - dataset to use when training a model without the images
2. ImagesAndExtraFeaturesOnly - dataset to use when training a model with the images
"""
from torch.utils.data import Dataset

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
  extra_features: Extra features that are to be used when training the model. Defatult = None.
  """
  def __init__(self, paths, transform, label, datetime_index, ghi_cs, extra_features=None):
    self.paths = paths
    self.transform = transform
    self.label = label
    self.extra_features = extra_features
    self.datetime_index = datetime_index 
    self.ghi_cs = ghi_cs 

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    image_path = self.paths.iloc[index]
    time_stamp = self.datetime_index[index] 
    current_ghi_cs = self.ghi_cs[index] 
    time_stamp = torch.tensor(np.array([time_stamp.timestamp()])) 
    img = read_image(image_path)
    img = self.transform(img)
    ghi = self.label[index]
    if self.extra_features != None:
      extra_features = self.extra_features[index]
      return img, torch.cat((time_stamp, current_ghi_cs, ghi, extra_features), dim=0)
    else:
      return img, torch.cat((time_stamp, current_ghi_cs, ghi), dim=0)