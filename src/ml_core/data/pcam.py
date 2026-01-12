from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None, filter_data: bool = False):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform
        self.filter_data = filter_data

        # 1. Check existence
        if not self.x_path.exists():
            raise FileNotFoundError(f"File not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"File not found: {self.y_path}")

        # 2. Lazy Loading setup
        self.x_db = None
        self.y_db = None

        # 3. Get total length and optionally filter
        with h5py.File(self.x_path, 'r') as f:
            total_len = len(f['x'])
            
            if self.filter_data:
                # Heuristic: Scan for images that are not purely black (0) or white (255)
                # Note: In a real massive dataset, we would cache this index list.
                # For the assignment, we scan if requested (mostly by tests).
                self.indices = []
                data = f['x']
                for i in range(total_len):
                    # Quick check: Mean value of the image
                    # We accept images that are NOT extremely dark (<5) or bright (>250)
                    # We read just the mean to save memory
                    img_mean = data[i].mean()
                    if 5 < img_mean < 250:
                        self.indices.append(i)
            else:
                # If no filter, use all indices
                self.indices = list(range(total_len))

        self._len = len(self.indices)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map the requested index to the REAL index (skipping bad data)
        real_idx = self.indices[idx]

        # Lazy Open
        if self.x_db is None:
            self.x_db = h5py.File(self.x_path, 'r')
            self.y_db = h5py.File(self.y_path, 'r')

        # Read specific sample
        img = self.x_db['x'][real_idx]
        label = self.y_db['y'][real_idx].flatten()[0]

        # Preprocessing
        img = img.astype('float32') / 255.0
        img = np.clip(img, 0.0, 1.0)

        # To Tensor
        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label
