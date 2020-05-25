import torchvision
from torchvision import datasets

import torch
from torch.utils.data import Dataset, DataLoader

from albumentations import Compose, Rotate, HorizontalFlip, Normalize, RandomCrop, Cutout, PadIfNeeded, Lambda
from albumentations.pytorch import ToTensor

import cv2
import os
import glob

import pandas as pd
import numpy as np

import subprocess
import urllib

from multiprocessing import Process
from zipfile import ZipFile, ZIP_DEFLATED

from torchvision.datasets.utils import download_url, download_and_extract_archive

class FgBgDataset(Dataset):
  """
  Custom class to load foreground background image dataset
  """
  def __init__(self, no_of_parts=10, 
               fg_bg_train_transform=None, 
               bg_train_transform=None, 
               fg_bg_test_transform=None, 
               bg_test_transform=None,
               target_transform=None, 
               mask_transform = None,
               depth_transform = None,
               **kwargs):
    """
    Constructor for foreground background image dataset
    """
    super().__init__(**kwargs)

    self.ROOT_URL = 'https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/'
    self.COMPRESSED_DIR = './compressed'
    self.DATASET_DIR = './dataset'
    self.DATASET_INFO_FILE = 'dataset_info.csv'
    self.DATASET_FILE_PREFIX = 'dataset_'

    self.no_of_parts = no_of_parts
    self.dataset_info_file_path = os.path.join(self.DATASET_DIR, self.DATASET_INFO_FILE)

    self.download_dataset()

    self.df = pd.read_csv(self.dataset_info_file_path)

    self.train = True
    self.fg_bg_train_transform = fg_bg_train_transform 
    self.bg_train_transform = bg_train_transform 
    self.fg_bg_test_transform = fg_bg_test_transform 
    self.bg_test_transform = bg_test_transform

    self.mask_transform = mask_transform
    self.depth_transform = depth_transform

    self.is_random_crop = False

  def download_dataset(self):
    if not os.path.exists(self.DATASET_DIR):
      print('Downloading dataset..')
      os.mkdir(self.DATASET_DIR)
      if not os.path.exists(self.COMPRESSED_DIR):
        os.mkdir(self.COMPRESSED_DIR)

      dataset_info_url = urllib.parse.urljoin(self.ROOT_URL, self.DATASET_INFO_FILE)
      download_url(dataset_info_url, self.DATASET_DIR, self.DATASET_INFO_FILE)
      dataset_info_url = urllib.parse.urljoin(self.ROOT_URL, self.DATASET_INFO_FILE)

      processes = [Process(target=self.download_and_extract_part, args=(part_idx,)) for part_idx in range(self.no_of_parts)]
      processes.append(Process(target=self.download_and_extract_bg))
      for process in processes:
        process.start()
      for process in processes:
        process.join()
    else:
      print('Dataset found!')

  def download_and_extract_part(self, part):
    dataset_part_path = f'{self.DATASET_FILE_PREFIX}{part}.zip'
    dataset_part_url = urllib.parse.urljoin(self.ROOT_URL, dataset_part_path)
    download_and_extract_archive(dataset_part_url, self.COMPRESSED_DIR, self.DATASET_DIR)
  
  def download_and_extract_bg(self):
    bg_path = f'bg_cropped.zip'
    bg_url = urllib.parse.urljoin(self.ROOT_URL, bg_path)
    download_and_extract_archive(bg_url, self.COMPRESSED_DIR, self.DATASET_DIR)

  def __len__(self):
      return len(self.df)

  def set_train(self):
      self.train = True

  def set_eval(self):
      self.train = False

  def __getitem__(self, index):

    H = 1024
    W = 1024

    required_height = 256
    required_width = 256

    if torch.is_tensor(index):
      index = index.tolist()

    fg_bg_path = os.path.join(self.DATASET_DIR, self.df['fg_bg_paths'].iloc[index])
    bg_path = os.path.join(self.DATASET_DIR, self.df['selected_bg_paths'].iloc[index])
    mask_path = os.path.join(self.DATASET_DIR, self.df['mask_paths'].iloc[index])
    depth_map_path = os.path.join(self.DATASET_DIR, self.df['depth_map_paths'].iloc[index])

    fg_bg_img = cv2.imread(fg_bg_path)
    fg_bg_img = cv2.cvtColor(fg_bg_img, cv2.COLOR_BGR2RGB)

    bg_img = cv2.imread(bg_path)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask_img = cv2.threshold(mask_img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
    depth_img = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    if self.is_random_crop:
      r_min = 0
      r_max = H - required_height

      c_min = 0
      c_max = W - required_width

      if self.train:
        r = np.random.randint(r_min, r_max)
        c = np.random.randint(c_min, c_max)
      else:
        r = 0
        c = 0

      fg_bg_img = fg_bg_img[r:r + required_height, c:c + required_width, :]
      bg_img = bg_img[r:r + required_height, c:c + required_width, :]
      mask_img = mask_img[r:r + required_height, c:c + required_width]
      depth_img = depth_img[(r // 2):(r + required_height) // 2, (c // 2):(c + required_width) // 2]
    else:
      fg_bg_img = cv2.resize(fg_bg_img, (required_height, required_height), cv2.INTER_CUBIC)
      bg_img = cv2.resize(bg_img, (required_height, required_height), cv2.INTER_CUBIC)

    mask_minimal = cv2.resize(mask_img, (128, 128), cv2.INTER_CUBIC)
    depth_minimal = cv2.resize(depth_img, (128, 128), cv2.INTER_CUBIC)

    mask_img = np.expand_dims(mask_img, axis=2)
    depth_img = np.expand_dims(depth_img, axis=2)
    
    mask_minimal = np.expand_dims(mask_minimal, axis=2)
    depth_minimal = np.expand_dims(depth_minimal, axis=2)

    is_camel = (mask_img > 128).flatten().any()
    # print(fg_bg_img.shape, bg_img.shape, mask_img.shape, depth_img.shape)

    if self.train:
      if self.fg_bg_train_transform:
        augmented = self.fg_bg_train_transform(image=fg_bg_img)
        fg_bg_img = augmented['image']
      if self.bg_train_transform:
        augmented = self.bg_train_transform(image=bg_img)
        bg_img = augmented['image']
    else:
      if self.fg_bg_test_transform:
        augmented = self.fg_bg_test_transform(image=fg_bg_img)
        fg_bg_img = augmented['image']
      if self.bg_test_transform:
        augmented = self.bg_test_transform(image=bg_img)
        bg_img = augmented['image']

    if self.depth_transform is not None:
        # augmented = self.depth_transform(image=depth_img)
        # depth_img = augmented['image']

        augmented = self.depth_transform(image=depth_minimal)
        depth_minimal = augmented['image']

    if self.mask_transform is not None:
        # augmented = self.mask_transform(image=mask_img)
        # mask_img = augmented['image']

        augmented = self.mask_transform(image=mask_minimal)
        mask_minimal = augmented['image']
    
    return { 'fg_bg': fg_bg_img, 'mask': mask_img, 'depth': depth_img, 'bg': bg_img, 
             'is_camel': is_camel, 'mask_minimal': mask_minimal, 'depth_minimal': depth_minimal
            }

class TinyImageNet(Dataset):
    """
    Custom class to load TinyImagenet
    """

    def __init__(self, root_dir='/content/tiny-imagenet-200', train_transform=None, test_transform=None, target_transform=None, **kwargs):
        """
        Constructor for Tiny Imagenet dataset
        """
        super().__init__(**kwargs)
        self.root_dir = root_dir

        # Load all file paths
        self.wnids_file = os.path.join(self.root_dir, 'wnids.txt')
        self.words_file = os.path.join(self.root_dir, 'words.txt')
        self.val_file = os.path.join(self.root_dir, 'val', 'val_annotations.txt')
        self.train_folder = os.path.join(self.root_dir, 'train')
        self.val_folder = os.path.join(self.root_dir, 'val')

        # Creating dataframes to hold data
        self.wnids_df = pd.read_table(self.wnids_file, header=None, names=['id'])
        self.words_df = pd.read_table(
            self.words_file, header=None, names=['id', 'name'])
        self.wnids_df = pd.merge(self.wnids_df, self.words_df, on=['id'])

        self.val_df = pd.read_table(self.val_file, header=None, names=[
                                    'path', 'id', 'c_x', 'c_y', 'w', 'h'])
        self.val_df = pd.merge(self.val_df, self.wnids_df, on=['id'])

        data_items = []

        # Load data from train folder
        for idx, row in self.wnids_df.iterrows():
            image_id = row['id']
            name = row['name']
            paths = glob.glob(f'{self.train_folder}/{image_id}/images/{image_id}_*')
            for path in paths:
                data_items.append({'id': image_id, 'name': name, 'path': path})

        # Load data from val folder
        for idx, row in self.val_df.iterrows():
            image_id = row['id']
            name = row['name']
            path = os.path.join(self.val_folder, 'images', row['path'])
            data_items.append({'id': image_id, 'name': name, 'path': path})

        # Create mixed data
        self.df = pd.DataFrame(data_items)
        labels = self.df['id'].unique().tolist()
        self.label_map = dict(zip(labels, np.arange(len(labels))))
        self.df['target'] = self.df['id'].map(self.label_map)

        # Default to train mode
        self.train = True
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        path = self.df['path'].iloc[index]
        target = self.df['target'].iloc[index]

        # print(index, path, target)

        img = cv2.imread(path)

        # print(img.shape)

        if self.train:
            if self.train_transform:
                augmented = self.train_transform(image=img)
                img = augmented['image']
        else:
            if self.test_transform:
                augmented = self.test_transform(image=img)
                img = augmented['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_dataset(datasets.CIFAR10):
    """
    Custom class to include albumentations data augmentations
    """

    def __init__(self, **kwargs):
        """
        Constructor for custom CIFAR10 dataset
        """
        super().__init__(**kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class DataManager(object):
    """
    Class that handles data management for an experiment
    """

    def __init__(self, batch_size=64, use_cuda=True, dataset_name='cifar10',
                 trainset=None, testset=None,
                 train_transforms=None, test_transforms=None):
        super().__init__()

        self.dataset_name = dataset_name
        self.batch_size = batch_size

        if self.dataset_name == 'cifar10':
            if trainset:
                self.trainset = trainset
            else:
                # Train Phase transformations
                if train_transforms:
                    self.train_transforms = train_transforms
                else:
                    self.train_transforms = Compose([Lambda(image=DataManager.padded_random_crop, always_apply=False, p=0.5),
                                                                        Cutout(
                        num_holes=4, max_h_size=8, max_w_size=8, always_apply=False, p=0.5),
                        HorizontalFlip(
                        p=0.5),
                        Normalize(
                                                                            mean=[
                                                                                0.4914, 0.4822, 0.4465],
                                                                            std=[
                                                                                0.2471, 0.2435, 0.2616],
                    ),
                        ToTensor()
                    ])
                self.trainset = CIFAR10_dataset(
                    root='./data', train=True, download=True, transform=self.train_transforms)

            if testset:
                self.testset = testset
            else:
                # Test Phase transformations
                if test_transforms:
                    self.test_transforms = test_transforms
                else:
                    self.test_transforms = Compose([Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2471, 0.2435, 0.2616],
                    ),
                        ToTensor()
                    ])
                self.testset = CIFAR10_dataset(
                    root='./data', train=False, download=True, transform=self.test_transforms)

            # dataloader arguments - something you'll fetch these from cmdprmt
            dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4,
                                   pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=batch_size)

            # train dataloader
            self.train_loader = torch.utils.data.DataLoader(
                self.trainset, **dataloader_args)
            # test dataloader
            self.test_loader = torch.utils.data.DataLoader(
                self.testset, **dataloader_args)

        elif dataset_name == 'tiny_imagenet':
            if trainset:
                self.trainset = trainset
            else:
                # Train Phase transformations
                if train_transforms:
                    self.train_transforms = train_transforms
                else:
                    self.train_transforms = Compose([
                                                    Lambda(image=DataManager.padded_random_crop_tiny_imagenet, always_apply=False, p=0.75),
                                                    Cutout(num_holes=8, max_h_size=16, max_w_size=16, always_apply=False, p=0.75),
                                                    HorizontalFlip(p=0.5),
                                                    Normalize(
                                                        mean=[0.39755616, 0.44819888, 0.48042724],
                                                        std=[0.28166872, 0.26888603, 0.2764395],
                                                    ),
                                                    ToTensor()
                                                ])
            if testset:
                self.testset = testset
            else:
                # Test Phase transformations
                if test_transforms:
                    self.test_transforms = test_transforms
                else:
                    self.test_transforms = Compose([Normalize(
                                                        mean=[0.39755616, 0.44819888, 0.48042724],
                                                        std=[0.28166872, 0.26888603, 0.2764395],
                                                    ),
                                                    ToTensor()
                                                    ])

            self.dataset = TinyImageNet(train_transform=self.train_transforms, test_transform=self.test_transforms)

            validation_split = 0.7
            train_count = int(validation_split * len(self.dataset))
            validation_count = int((1 - validation_split) * len(self.dataset))

            print(f'Found {train_count} train images, {validation_count} validation images')

            self.trainset, self.testset = torch.utils.data.random_split(self.dataset, [train_count, validation_count])

            # dataloader arguments - something you'll fetch these from cmdprmt
            dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4,
                                   pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=batch_size)

            # train dataloader
            self.train_loader = torch.utils.data.DataLoader(
                self.trainset, **dataloader_args)
            # test dataloader
            self.test_loader = torch.utils.data.DataLoader(
                self.testset, **dataloader_args)

        elif dataset_name == 'camel_dataset':
            self.fg_bg_train_transform = Compose([Normalize(
                                                mean=[0.4561, 0.3984, 0.3620],
                                                std=[0.2866, 0.2640, 0.2774],
                                            ),
                                            ToTensor()
                                        ])
            self.bg_train_transform = Compose([Normalize(
                                                mean=[0.4547, 0.4028, 0.3705],
                                                std=[0.2920, 0.2662, 0.2803],
                                            ),
                                            ToTensor()
                                        ])
            self.mask_transform = Compose([ToTensor()])
            self.depth_transform = Compose([Normalize(
                                                mean=[0.6289],
                                                std=[0.2238],
                                            ),
                                    ToTensor()])
            self.fg_bg_test_transform = Compose([ToTensor()])
            self.bg_test_transform = Compose([ToTensor()])
            self.dataset = FgBgDataset(fg_bg_train_transform=self.fg_bg_train_transform,
                      bg_train_transform=self.bg_train_transform,
                      fg_bg_test_transform=self.fg_bg_test_transform, 
                      bg_test_transform=self.bg_test_transform, 
                      mask_transform=self.mask_transform, 
                      depth_transform=self.depth_transform)

            validation_split = 0.75
            train_count = int(validation_split * len(self.dataset))
            validation_count = int((1 - validation_split) * len(self.dataset))

            print(f'Found {train_count} train images, {validation_count} validation images')

            self.trainset, self.testset = torch.utils.data.random_split(self.dataset, [train_count, validation_count])

            # dataloader arguments - something you'll fetch these from cmdprmt
            dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4,
                                   pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=batch_size)

            # train dataloader
            self.train_loader = torch.utils.data.DataLoader(
                self.trainset, **dataloader_args)
            # test dataloader
            self.test_loader = torch.utils.data.DataLoader(
                self.testset, **dataloader_args)

    @staticmethod
    def padded_random_crop(x, **kwargs):
        x = PadIfNeeded(min_height=40, min_width=40, border_mode=4,
                        value=None, mask_value=None, always_apply=True).apply(x)
        x = RandomCrop(height=32, width=32, always_apply=True).apply(x)
        return x
    
    @staticmethod
    def padded_random_crop_tiny_imagenet(x, **kwargs):
        x = PadIfNeeded(min_height=72, min_width=72, border_mode=4,
                        value=None, mask_value=None, always_apply=True).apply(x)
        x = RandomCrop(height=64, width=64, always_apply=True).apply(x)
        return x

    def set_train(self):
        if self.dataset_name == 'tiny_imagenet':
            self.dataset.set_train()
    
    def set_eval(self):
        if self.dataset_name == 'tiny_imagenet':
            self.dataset.set_eval()