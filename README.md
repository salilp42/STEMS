# STEMS: Segmented Time-series End-to-end Multi-scale/Multimodal System

A state-of-the-art pipeline for time-series analysis combining TabNet, temporal cross-attention, and advanced segmentation techniques.

## Research Focus

### Mathematical Framework
- Optimal segmentation length analysis using wavelet coherence
- Bounds on aggregation error propagation
- Information-theoretic analysis of segment overlap
- Convergence proofs for aggregation strategies
- Uncertainty quantification in temporal aggregation

### Experimental Analysis
- Segment length optimization studies
- Comparative analysis of aggregation methods:
  - Attention-weighted voting
  - Hierarchical clustering
  - Optimal transport
  - Uncertainty-based weighting
- Ablation studies on feature importance
- Statistical significance testing

## Mathematical Experiments

Run segmentation experiments:
```python
from stems.experiments import run_segment_analysis

# Analyze optimal segment lengths
results = run_segment_analysis(
    data,
    lengths=[50, 100, 200, 500],
    overlaps=[0, 0.25, 0.5, 0.75]
)

# Compare aggregation strategies
agg_comparison = compare_aggregators(
    data,
    methods=['attention', 'hierarchical', 'ot'],
    metrics=['accuracy', 'f1', 'temporal_consistency']
)

## Features

- Full TabNet with feature masking
- Temporal cross-attention
- Temporal pyramid pooling (TPP)
- Variational layers for uncertainty
- Causal dilated convolutions
- Multiple aggregation strategies
- OOD detection
- Interpretability tools

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```python
from stems.model import STEMSModel
from stems.data import TimeSeriesDataset

# Initialize model
model = STEMSModel()

# Load data
dataset = TimeSeriesDataset(raw_data, labels)

# Train
model.fit(dataset)
```

## Project Structure

```
STEMS/
├── stems/                      # Main package directory
│   ├── __init__.py
│   ├── model.py               # Core STEMS model
│   ├── data.py                # Dataset and data loading
│   ├── attention.py           # Attention mechanisms
│   ├── aggregation.py         # Aggregation strategies
│   ├── interpretability.py    # Interpretability tools
│   └── utils.py               # Utility functions
├── tests/                     # Test directory
├── notebooks/                 # Jupyter notebooks
├── requirements.txt           # Dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## License

MIT License - see LICENSE file for details

## Author

Salil Patel
