"""
Dataset and data loading utilities for STEMS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Union


class LearnedSegmentLength(nn.Module):
    """Neural module to learn optimal segment length."""
    
    def __init__(self, min_length: int = 50, max_length: int = 2000):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = self.fc(x)
        length = self.min_length + (self.max_length - self.min_length) * scaled
        return length


def segment_time_series(
    time_series: np.ndarray,
    seg_length: int,
    stride: Optional[int] = None
) -> List[np.ndarray]:
    """
    Segment time series into overlapping or non-overlapping segments.
    
    Args:
        time_series: Input time series of shape (C, T)
        seg_length: Length of each segment
        stride: Stride between segments. If None, use seg_length
    
    Returns:
        List of segments, each of shape (C, seg_length)
    """
    if stride is None:
        stride = seg_length
    
    segments = []
    T = time_series.shape[-1]
    start = 0
    
    while (start + seg_length) <= T:
        segments.append(time_series[..., start:start+seg_length])
        start += stride
    
    return segments


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data with dynamic or static segmentation.
    
    Args:
        raw_data: List of time series, each of shape (C, T)
        labels: List of corresponding labels
        learn_length: Whether to learn segment length
        transform: Optional transform to apply to segments
    """
    
    def __init__(
        self,
        raw_data: List[np.ndarray],
        labels: List[int],
        learn_length: bool = False,
        transform: Optional[callable] = None
    ):
        self.raw_data = raw_data
        self.labels = labels
        self.transform = transform
        self.learn_length = learn_length
        
        if learn_length:
            self.length_model = LearnedSegmentLength()
        
        # Initialize with static segmentation
        self.segments = []
        self.segment_labels = []
        self._init_segments()
    
    def _init_segments(self, seg_length: int = 100):
        """Initialize segments with given length."""
        stride = seg_length // 2
        
        for x, y in zip(self.raw_data, self.labels):
            segs = segment_time_series(x, seg_length, stride)
            self.segments.extend([torch.tensor(s, dtype=torch.float32) for s in segs])
            self.segment_labels.extend([y] * len(segs))
    
    def update_segmentation(self, new_length: int):
        """Update segmentation with new segment length."""
        self.segments = []
        self.segment_labels = []
        self._init_segments(new_length)
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self.segments[idx]
        y = self.segment_labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def create_dataloader(
    dataset: TimeSeriesDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    dynamic_batching: bool = False
) -> DataLoader:
    """
    Create a DataLoader with optional dynamic batching.
    
    Args:
        dataset: TimeSeriesDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        dynamic_batching: Whether to use dynamic batching
    
    Returns:
        DataLoader instance
    """
    if dynamic_batching:
        # Custom collate function for dynamic batching
        def collate_fn(batch):
            # Sort by sequence length
            batch.sort(key=lambda x: x[0].shape[-1], reverse=True)
            xs, ys = zip(*batch)
            
            # Pad to max length in batch
            max_len = xs[0].shape[-1]
            xs_padded = []
            
            for x in xs:
                pad_len = max_len - x.shape[-1]
                if pad_len > 0:
                    x = F.pad(x, (0, pad_len))
                xs_padded.append(x)
            
            x_batch = torch.stack(xs_padded)
            y_batch = torch.tensor(ys, dtype=torch.long)
            
            return x_batch, y_batch
    else:
        collate_fn = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
