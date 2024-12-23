"""
Interpretability tools for STEMS models.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable
from sklearn.covariance import EmpiricalCovariance

try:
    from captum.attr import (
        IntegratedGradients,
        DeepLift,
        GradientShap,
        Occlusion
    )
except ImportError:
    IntegratedGradients = None
    DeepLift = None
    GradientShap = None
    Occlusion = None


class ModelWrapper:
    """Wrapper to make model compatible with interpretability tools."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits


class MahalanobisOOD:
    """
    Out-of-distribution detection using Mahalanobis distance.
    
    Args:
        support_fraction: Fraction of points to use for covariance estimation
    """
    
    def __init__(self, support_fraction: float = 1.0):
        self.cov = EmpiricalCovariance(support_fraction=support_fraction)
        self.fitted = False
        self.mean = None
    
    def fit(self, embeddings: np.ndarray):
        """
        Fit the covariance model on in-distribution embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
        """
        self.cov.fit(embeddings)
        self.mean = np.mean(embeddings, axis=0)
        self.fitted = True
    
    def score(self, embedding: np.ndarray) -> float:
        """
        Compute Mahalanobis distance score for a sample.
        
        Args:
            embedding: Array of shape (n_features,)
        
        Returns:
            Mahalanobis distance score
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before scoring")
        
        embedding = embedding.reshape(1, -1)
        return self.cov.mahalanobis(embedding)[0]


def compute_integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: int,
    n_steps: int = 50,
    internal_batch_size: Optional[int] = None
) -> Tuple[torch.Tensor, float]:
    """
    Compute integrated gradients attribution.
    
    Args:
        model: The model
        inputs: Input tensor
        target: Target class
        n_steps: Number of steps in integral approximation
        internal_batch_size: Batch size for internal processing
    
    Returns:
        attributions: Attribution scores
        delta: Approximation error
    """
    if IntegratedGradients is None:
        raise ImportError("Captum is required for integrated gradients")
    
    wrapped_model = ModelWrapper(model)
    ig = IntegratedGradients(wrapped_model)
    
    attr, delta = ig.attribute(
        inputs,
        target=target,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True
    )
    
    return attr, delta


def compute_deeplift(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: int,
    baselines: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute DeepLIFT attribution scores.
    
    Args:
        model: The model
        inputs: Input tensor
        target: Target class
        baselines: Baseline inputs
    
    Returns:
        Attribution scores
    """
    if DeepLift is None:
        raise ImportError("Captum is required for DeepLIFT")
    
    wrapped_model = ModelWrapper(model)
    dl = DeepLift(wrapped_model)
    
    attributions = dl.attribute(inputs, target=target, baselines=baselines)
    return attributions


def compute_gradient_shap(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: int,
    n_samples: int = 50
) -> torch.Tensor:
    """
    Compute GradientSHAP attribution scores.
    
    Args:
        model: The model
        inputs: Input tensor
        target: Target class
        n_samples: Number of samples
    
    Returns:
        Attribution scores
    """
    if GradientShap is None:
        raise ImportError("Captum is required for GradientSHAP")
    
    wrapped_model = ModelWrapper(model)
    gs = GradientShap(wrapped_model)
    
    # Create baseline distribution
    baselines = torch.randn_like(inputs).repeat(n_samples, 1, 1)
    
    attributions = gs.attribute(inputs, baselines=baselines, target=target)
    return attributions


def occlusion_sensitivity(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: int,
    window_size: int = 10,
    stride: int = 5
) -> torch.Tensor:
    """
    Compute occlusion sensitivity maps.
    
    Args:
        model: The model
        inputs: Input tensor
        target: Target class
        window_size: Size of occlusion window
        stride: Stride for sliding window
    
    Returns:
        Sensitivity scores
    """
    if Occlusion is None:
        raise ImportError("Captum is required for occlusion sensitivity")
    
    wrapped_model = ModelWrapper(model)
    occlusion = Occlusion(wrapped_model)
    
    attributions = occlusion.attribute(
        inputs,
        target=target,
        sliding_window_shapes=(1, window_size),
        strides=(1, stride)
    )
    
    return attributions
