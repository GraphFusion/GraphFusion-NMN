import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.confidence_scorer import ConfidenceScorer
import torch
import pytest

def test_confidence_scorer():
    scorer = ConfidenceScorer(hidden_size=10)
    memory_output = torch.randn(5, 10)  # 5 samples, hidden size 10
    confidence = scorer(memory_output)
    assert confidence.shape == (5, 1)  # Output should be (batch_size, 1)
    assert confidence.min() >= 0 and confidence.max() <= 1  # Confidence is between 0 and 1
