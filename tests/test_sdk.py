import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sdk.graphfusion import GraphFusion
import torch
import pytest

def test_graphfusion_process():
    graphfusion = GraphFusion(input_size=10, hidden_size=20)
    input_data = torch.randn(5, 10)  # 5 samples, input size 10
    result = graphfusion.process_input(input_data)
    assert 'output' in result
    assert 'confidence' in result
    assert 'cell_id' in result

def test_graphfusion_query():
    graphfusion = GraphFusion(input_size=10, hidden_size=20)
    query_vector = torch.randn(10)  # Query vector
    results = graphfusion.query(query_vector, top_k=3)
    assert len(results) == 3  # Ensure we get top 3 results
    assert 'node_id' in results[0]
    assert 'similarity' in results[0]
