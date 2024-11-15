import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.neural_memory import NeuralMemoryNetwork
import torch
import pytest

def test_process_input():
    network = NeuralMemoryNetwork(input_size=10, hidden_size=20)
    input_data = torch.randn(5, 10)  # 5 samples, input size 10
    context = {'context_key': 'value'}
    result = network.process(input_data, context)
    assert 'output' in result
    assert 'confidence' in result
    assert 'cell_id' in result

def test_query():
    network = NeuralMemoryNetwork(input_size=10, hidden_size=20)
    query_vector = torch.randn(10)  # Query vector, same size as input data
    results = network.query(query_vector)
    assert len(results) > 0  # Ensure we get some results
    assert 'node_id' in results[0]
    assert 'similarity' in results[0]
