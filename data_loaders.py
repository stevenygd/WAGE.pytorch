import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
import tabulate
import models
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import json


def get_data_loaders(dataset, data_path, val_ratio, batch_size, num_workers):
    if dataset=="CIFAR10":
        ds = getattr(datasets, dataset)
        path = os.path.join(data_path, dataset.lower())
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = ds(path, train=True, download=True, transform=transform_train)
        val_set = ds(path, train=True, download=True, transform=transform_test)
        test_set = ds(path, train=False, download=True, transform=transform_test)
        if val_ratio != 0:
            train_size = len(train_set)
            indices = list(range(train_size))
            val_size = int(val_ratio*train_size)
            print("train set size {}, validation set size {}".format(train_size-val_size, val_size))
            np.random.shuffle(indices)
            val_idx, train_idx = indices[train_size-val_size:], indices[:train_size-val_size]
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
        else :
            train_sampler = None
            val_sampler = None
        num_classes = 10

    loaders = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    return loaders

