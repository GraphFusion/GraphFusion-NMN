import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
from typing import Dict, Any, Optional
import json

class KnowledgeGraph:
    """
    Dynamic knowledge graph with confidence tracking.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_node(self, 
                 node_id: str, 
                 features: torch.Tensor, 
                 metadata: Dict[str, Any],
                 confidence: float) -> None:
        """Add a node to the graph with features and metadata."""
        self.graph.add_node(
            node_id,
            features=features.detach().numpy(),
            metadata=metadata,
            confidence=confidence
        )
    
    def add_edge(self, 
                 source_id: str, 
                 target_id: str, 
                 relationship_type: str,
                 confidence: float) -> None:
        """Add edge between nodes with relationship type."""
        self.graph.add_edge(
            source_id,
            target_id,
            type=relationship_type,
            confidence=confidence
        )
    
    def export(self, format: str = 'json') -> str:
        """Export the graph in the specified format."""
        if format == 'json':
            return json.dumps(nx.node_link_data(self.graph))
        raise ValueError(f"Unsupported format: {format}")
