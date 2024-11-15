import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.knowledge_graph import KnowledgeGraph
import torch
import pytest
def test_add_node():
    graph = KnowledgeGraph()
    node_data = {
        'features': torch.randn(10),
        'metadata': {'type': 'entity'},
        'confidence': 0.9
    }
    graph.add_node("node_1", **node_data)
    assert "node_1" in graph.graph.nodes
    assert graph.graph.nodes["node_1"]['confidence'] == 0.9

def test_add_edge():
    graph = KnowledgeGraph()
    graph.add_node("node_1", features=torch.randn(10), metadata={}, confidence=0.9)
    graph.add_node("node_2", features=torch.randn(10), metadata={}, confidence=0.9)
    graph.add_edge("node_1", "node_2", relationship_type="related", confidence=0.8)
    assert graph.graph.has_edge("node_1", "node_2")
    assert graph.graph["node_1"]["node_2"]['confidence'] == 0.8
