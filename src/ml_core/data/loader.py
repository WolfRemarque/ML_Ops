from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    val_transform = None

    train_x_path = base_path / "camelyonpatch_level_2_split_train_x.h5"
    train_y_path = base_path / "camelyonpatch_level_2_split_train_y.h5"
    val_x_path = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    val_y_path = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    train_dataset = PCAMDataset(str(train_x_path), str(train_y_path), transform=train_transform)
    val_dataset = PCAMDataset(str(val_x_path), str(val_y_path), transform=val_transform)

    with h5py.File(train_y_path, 'r') as f:
        labels = np.array(f['y']).flatten()

    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
