import pytest
import torch
import numpy as np

from stems.model import STEMSModel
from stems.data import TimeSeriesDataset, segment_time_series
from stems.attention import TemporalCrossAttention
from stems.aggregation import attention_weighted_voting
from stems.utils import set_seed


@pytest.fixture
def model():
    set_seed(42)
    return STEMSModel()


@pytest.fixture
def sample_data():
    # Create synthetic data
    n_samples = 10
    time_steps = 100
    n_channels = 1
    
    data = []
    labels = []
    
    for _ in range(n_samples):
        x = np.random.randn(n_channels, time_steps)
        y = np.random.randint(0, 2)
        data.append(x)
        labels.append(y)
    
    return data, labels


def test_model_forward(model):
    batch_size = 2
    time_steps = 100
    x = torch.randn(batch_size, 1, time_steps)
    
    logits, kl_div = model(x)
    
    assert logits.shape == (batch_size, model.config.N_CLASSES)
    assert isinstance(kl_div, torch.Tensor)
    assert kl_div.ndim == 0  # scalar


def test_dataset(sample_data):
    data, labels = sample_data
    dataset = TimeSeriesDataset(data, labels)
    
    assert len(dataset) > 0
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)


def test_segmentation(sample_data):
    data, _ = sample_data
    x = data[0]  # (channels, time_steps)
    
    seg_length = 50
    segments = segment_time_series(x, seg_length)
    
    assert len(segments) > 0
    assert all(seg.shape[-1] == seg_length for seg in segments)


def test_attention():
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    attention = TemporalCrossAttention(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = attention(x, x, x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape[-2:] == (seq_len, seq_len)


def test_aggregation():
    n_segments = 5
    n_classes = 2
    segment_logits = [torch.randn(n_classes) for _ in range(n_segments)]
    
    aggregated = attention_weighted_voting(segment_logits)
    
    assert aggregated.shape == (n_classes,)
