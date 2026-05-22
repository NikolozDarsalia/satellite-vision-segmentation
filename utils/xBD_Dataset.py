import os
import glob
import imageio
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
  imgarr = np.asarray(img)
  proc_img = np.empty_like(imgarr, np.float32)

  proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
  proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
  proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
  return proc_img

class xBD_Dataset(BaseDataset):
  """xBD Dataset, Read images, apply augmentation and preprocessing transformations
  Args:
    dataset_path(str): the root path of the xBD Dataset
    type(str): train, test or hold, choose the folder you need to load
  """
  def __init__(self, dataset_path, type='train', augmentation=None, preprocessing=None):
    self.dataset_path = dataset_path
    self.type = type
    self.augmentation = augmentation
    self.preprocessing = preprocessing
    filenames = list(set([filename.split('_p')[0] for filename in os.listdir(os.path.join(dataset_path, type, 'images'))]))
    self.pre_image_list = [os.path.join(dataset_path,type,"images",filename+"_pre_disaster.tif") for filename in filenames]
    self.post_image_list = [os.path.join(dataset_path,type,"images",filename+"_post_disaster.tif") for filename in filenames]
    self.building_mask_list = [os.path.join(dataset_path,type,"masks",filename+"_pre_disaster.png") for filename in filenames]
    self.damage_mask_list = [os.path.join(dataset_path,type,"masks",filename+"_post_disaster.png") for filename in filenames]

  def __getitem__(self, index):
    pre_disaster_image_path = self.pre_image_list[index]
    post_disaster_image_path = self.post_image_list[index]
    building_footprint_label_path = self.building_mask_list[index]
    damage_label_path = self.damage_mask_list[index]
    pre_img = imageio.imread(pre_disaster_image_path).astype(np.float32)
    post_img = imageio.imread(post_disaster_image_path).astype(np.float32)
    building_label = imageio.imread(building_footprint_label_path).astype(np.float32)
    damage_label = imageio.imread(damage_label_path).astype(np.float32)

    if self.type == 'train':
      if self.augmentation:
        pre_img, post_img, building_label, damage_label = self.augmentation(pre_img, post_img, building_label, damage_label)
        
      if self.preprocessing:
        pre_img, post_img, building_label, damage_label = self.preprocessing(pre_img, post_img, building_label, damage_label)
        
    pre_img = normalize_img(pre_img) #imagenet normalization
    post_img = normalize_img(post_img)
    pre_img = np.transpose(pre_img, (2, 0, 1)) #pytorch requires channel, head, weight
    post_img = np.transpose(post_img, (2, 0, 1))

    building_label = np.asarray(building_label)
    damage_label = np.asarray(damage_label)

    return pre_img, post_img, building_label, damage_label

  def __len__(self):
    return len(self.pre_image_list)