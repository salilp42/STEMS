"""
Attention mechanisms for STEMS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism for temporal data.
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (n_heads, d_k)."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the heads back."""
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention."""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections and split heads
        q = self.split_heads(self.w_q(query))  # (batch_size, n_heads, seq_len, d_k)
        k = self.split_heads(self.w_k(key))
        v = self.split_heads(self.w_v(value))
        
        # Apply attention
        output, attention_weights = self.attention(q, k, v, mask)
        
        # Combine heads and apply layer norm
        output = self.combine_heads(output)
        output = self.layer_norm(output + query)  # Residual connection
        
        return output, attention_weights


class TemporalSelfAttention(TemporalCrossAttention):
    """Self-attention variant where query, key, and value are the same."""
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, x, x, mask)
