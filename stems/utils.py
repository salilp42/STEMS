"""
Utility functions for STEMS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple
import logging
from logging import handlers
import random


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(
    level: int = logging.INFO,
    filename: Optional[str] = "stems.log",
    max_bytes: int = 5_000_000,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        filename: Log file name
        max_bytes: Maximum size of log file
        backup_count: Number of backup files
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger("STEMS")
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if filename:
        file_handler = handlers.RotatingFileHandler(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def temporal_pyramid_pooling(
    x: torch.Tensor,
    levels: List[int] = [1, 2, 4]
) -> torch.Tensor:
    """
    Temporal pyramid pooling.
    
    Args:
        x: Input tensor of shape (batch_size, channels, time)
        levels: List of pooling levels
    
    Returns:
        Pooled features
    """
    batch_size = x.size(0)
    pooled_outputs = []
    
    for level in levels:
        # Apply adaptive average pooling
        kernel_size = max(x.size(-1) // level, 1)
        pool = nn.AdaptiveAvgPool1d(level)
        pooled = pool(x)
        pooled_outputs.append(pooled.view(batch_size, -1))
    
    # Concatenate all pooled features
    return torch.cat(pooled_outputs, dim=1)


def contrastive_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    NT-Xent (normalized temperature-scaled cross entropy) loss.
    
    Args:
        z_i: First set of embeddings
        z_j: Second set of embeddings
        temperature: Temperature parameter
    
    Returns:
        Contrastive loss value
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, dim)
    
    # Compute similarity matrix
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # (2B, 2B)
    
    # Mask for positive pairs
    pos_mask = torch.zeros_like(sim)
    pos_mask[torch.arange(batch_size), torch.arange(batch_size, 2*batch_size)] = 1
    pos_mask[torch.arange(batch_size, 2*batch_size), torch.arange(batch_size)] = 1
    
    # Scale by temperature
    sim = sim / temperature
    
    # Compute loss
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = -(log_prob * pos_mask).sum(dim=1).mean()
    
    return loss


def compute_segment_length_for_epoch(
    epoch: int,
    start_len: int = 100,
    end_len: int = 1000,
    num_steps: int = 5,
    total_epochs: int = 100
) -> int:
    """
    Compute segment length for curriculum learning.
    
    Args:
        epoch: Current epoch
        start_len: Initial segment length
        end_len: Final segment length
        num_steps: Number of curriculum steps
        total_epochs: Total number of epochs
    
    Returns:
        Segment length for current epoch
    """
    step_size = max(1, total_epochs // num_steps)
    progress = min(num_steps, epoch // step_size)
    length = start_len + (end_len - start_len) * (progress / num_steps)
    return int(length)


def evaluate_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Compute accuracy, precision, and recall.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
    
    Returns:
        accuracy, precision, recall
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    correct = (predictions == targets).sum()
    accuracy = correct / len(targets)
    
    # For binary classification
    if len(np.unique(targets)) == 2:
        true_pos = ((predictions == 1) & (targets == 1)).sum()
        pred_pos = (predictions == 1).sum()
        actual_pos = (targets == 1).sum()
        
        precision = true_pos / pred_pos if pred_pos > 0 else 0
        recall = true_pos / actual_pos if actual_pos > 0 else 0
    else:
        # For multi-class, use macro averaging
        precision = recall = 0
        classes = np.unique(targets)
        
        for c in classes:
            true_pos = ((predictions == c) & (targets == c)).sum()
            pred_pos = (predictions == c).sum()
            actual_pos = (targets == c).sum()
            
            precision += true_pos / pred_pos if pred_pos > 0 else 0
            recall += true_pos / actual_pos if actual_pos > 0 else 0
        
        precision /= len(classes)
        recall /= len(classes)
    
    return accuracy, precision, recall
