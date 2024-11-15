import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pytest
from core.memory_cell import MemoryCell

def test_memory_cell_initialization():
    cell = MemoryCell(input_size=10, hidden_size=20)
    assert isinstance(cell, MemoryCell)

def test_memory_cell_forward():
    cell = MemoryCell(input_size=10, hidden_size=20)
    input_data = torch.randn(5, 10)  # 5 samples, input size 10
    prev_memory = torch.zeros(20)    # Previous memory
    output, new_memory, confidence = cell(input_data, prev_memory)
    assert output.shape == (5, 20)  # Output should have shape (batch_size, hidden_size)
    assert isinstance(confidence, float)

