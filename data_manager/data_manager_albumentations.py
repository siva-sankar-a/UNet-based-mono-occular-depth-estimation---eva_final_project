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