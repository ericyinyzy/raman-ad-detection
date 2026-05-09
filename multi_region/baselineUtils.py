"""Data loading utilities for the multi-region (Raman-MRNet) pipeline.

Each `.npy` file under `../data/AD_processed_merge/mose2/{train,test}/{positive,negative}`
is a `(3, 701)` array stacking the cortex / hippocampus / thalamus Raman
spectra of one mouse sample. Positive samples are AD, negative samples are
healthy controls.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


DATA_ROOT = '../data/AD_processed_merge/mose2'


def create_pie_dataset(flag, device, mean=None, std=None):
    """Build an `OnboardTfDataset` for either the train or test split.

    The function preserves the original signature (mean/std are accepted but
    not applied) so existing callers keep working.
    """
    if flag == 'train':
        pos_data_dir = os.path.join(DATA_ROOT, 'train', 'positive')
        neg_data_dir = os.path.join(DATA_ROOT, 'train', 'negative')
    elif flag == 'test':
        pos_data_dir = os.path.join(DATA_ROOT, 'test', 'positive')
        neg_data_dir = os.path.join(DATA_ROOT, 'test', 'negative')
    else:
        raise ValueError(f"flag must be 'train' or 'test', got {flag!r}")

    data_list = get_data(pos_data_dir, neg_data_dir)
    data_array = np.stack([x[0] for x in data_list], axis=0)
    label_list = [x[1] for x in data_list]

    if flag == 'train':
        mean_value = data_array.mean(0)
        std_value = data_array.std(0)
        print(f'train data_array.shape = {data_array.shape}')
        data = [[data_array[i], label_list[i]] for i in range(data_array.shape[0])]
        return OnboardTfDataset(data, device), mean_value, std_value

    data = [[data_array[i], label_list[i]] for i in range(data_array.shape[0])]
    return OnboardTfDataset(data, device), None, None


def get_data(pos_dir, neg_dir):
    """Load every `.npy` under `pos_dir` (label=1) and `neg_dir` (label=0)."""
    data_list = []
    for file in os.listdir(pos_dir):
        data_list.append([np.load(os.path.join(pos_dir, file)), 1])
    for file in os.listdir(neg_dir):
        data_list.append([np.load(os.path.join(neg_dir, file)), 0])
    random.shuffle(data_list)
    return data_list


class OnboardTfDataset(Dataset):
    def __init__(self, data, device):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'input': torch.Tensor([self.data[index][0]]).to(self.device),
            'label': torch.Tensor([self.data[index][1]]).long().to(self.device),
        }
