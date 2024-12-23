"""
Aggregation strategies for combining segment predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from scipy.cluster.hierarchy import linkage, fcluster

try:
    import ot
except ImportError:
    ot = None


def attention_weighted_voting(
    segment_logits: List[torch.Tensor],
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Weighted sum of segment logits based on learned attention.
    
    Args:
        segment_logits: List of logits tensors, each of shape (n_classes,)
        temperature: Temperature for softmax
    
    Returns:
        Weighted sum of logits
    """
    stack = torch.stack(segment_logits, dim=0)  # (num_segments, n_classes)
    
    # Compute attention weights based on logit magnitudes
    weights = F.softmax(stack.norm(dim=1) / temperature, dim=0)
    weights = weights.unsqueeze(1)  # (num_segments, 1)
    
    weighted_sum = (weights * stack).sum(dim=0)
    return weighted_sum


def hierarchical_clustering_aggregator(
    segment_embeddings: List[torch.Tensor],
    cutoff: float = 0.5,
    method: str = 'ward'
) -> torch.Tensor:
    """
    Perform hierarchical clustering on segment embeddings.
    
    Args:
        segment_embeddings: List of embedding tensors
        cutoff: Distance threshold for clustering
        method: Linkage method ('ward', 'complete', 'average', etc.)
    
    Returns:
        Aggregated embedding
    """
    # Convert to numpy
    segs_np = torch.stack(segment_embeddings).cpu().numpy()
    
    # Compute linkage matrix
    linkage_matrix = linkage(segs_np, method=method)
    
    # Get cluster assignments
    cluster_labels = fcluster(linkage_matrix, t=cutoff, criterion='distance')
    
    # Aggregate within clusters
    unique_clusters = np.unique(cluster_labels)
    cluster_embeddings = []
    
    for c in unique_clusters:
        mask = (cluster_labels == c)
        cluster_mean = segs_np[mask].mean(axis=0)
        cluster_embeddings.append(cluster_mean)
    
    # Final aggregation across clusters
    aggregated = np.mean(cluster_embeddings, axis=0)
    return torch.tensor(aggregated, dtype=torch.float32)


def optimal_transport_aggregator(
    segment_probs: List[torch.Tensor],
    reg: float = 0.01,
    max_iter: int = 100
) -> torch.Tensor:
    """
    Aggregate probability distributions using optimal transport.
    
    Args:
        segment_probs: List of probability distributions
        reg: Entropic regularization parameter
        max_iter: Maximum number of Sinkhorn iterations
    
    Returns:
        Barycenter of the distributions
    """
    if ot is None:
        # Fallback to simple average if POT not installed
        return torch.mean(torch.stack(segment_probs), dim=0)
    
    # Convert to numpy
    P = torch.stack(segment_probs).cpu().numpy()
    
    # Uniform weights for distributions
    weights = np.ones(len(segment_probs)) / len(segment_probs)
    
    # Compute Wasserstein barycenter
    barycenter = ot.bregman.barycenter(
        P.T,
        weights,
        reg=reg,
        numItermax=max_iter
    )
    
    return torch.tensor(barycenter, dtype=torch.float32)


def temporal_uncertainty_propagation(
    segment_logits: List[torch.Tensor],
    uncertainties: List[float],
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Weight segments by their uncertainty estimates.
    
    Args:
        segment_logits: List of logits tensors
        uncertainties: List of uncertainty values (e.g., KL divergences)
        temperature: Temperature for softmax
    
    Returns:
        Uncertainty-weighted prediction
    """
    stack = torch.stack(segment_logits)  # (num_segments, n_classes)
    uncertainties = torch.tensor(uncertainties, dtype=torch.float32)
    
    # Convert uncertainties to weights (lower uncertainty -> higher weight)
    weights = torch.exp(-uncertainties / temperature)
    weights = weights / weights.sum()
    
    # Apply weights
    weighted_sum = (stack.T * weights).T.sum(dim=0)
    return weighted_sum
