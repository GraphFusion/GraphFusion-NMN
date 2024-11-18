import networkx as nx
import json
import torch
from typing import Dict, Any

class KnowledgeGraph:
    """
    Dynamic knowledge graph with confidence tracking.
    """
    def __init__(self):
        # Initialize an empty directed graph
        self.graph = nx.DiGraph()
    
    def add_node(self, 
                 node_id: str, 
                 features: torch.Tensor, 
                 metadata: Dict[str, Any],
                 confidence: float) -> None:
        """Add a node to the graph with features, metadata, and confidence."""
        # Add node data to the graph
        self.graph.add_node(
            node_id,
            features=features.detach().numpy(),  # Convert features tensor to numpy for storage
            metadata=metadata,
            confidence=confidence
        )
        print(f"Node added: {node_id}, Confidence: {confidence}")
    
    def add_edge(self, 
                 source_id: str, 
                 target_id: str, 
                 relationship_type: str,
                 confidence: float) -> None:
        """Add an edge between nodes with relationship type and confidence."""
        # Add edge data to the graph
        self.graph.add_edge(
            source_id,
            target_id,
            type=relationship_type,
            confidence=confidence
        )
        print(f"Edge added: {source_id} -> {target_id}, Confidence: {confidence}")
    
    def export(self, format: str = 'json') -> str:
        """Export the graph in the specified format."""
        if format == 'json':
            # Export the graph data to JSON format
            # Explicitly set edges="edges" to avoid FutureWarning in NetworkX 3.6
            return json.dumps(nx.node_link_data(self.graph, edges="edges"))
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_node_data(self, node_id: str) -> Dict[str, Any]:
        """Retrieve the data (features, metadata, and confidence) of a node by its ID."""
        if node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            return {
                "features": node_data["features"],
                "metadata": node_data["metadata"],
                "confidence": node_data["confidence"]
            }
        else:
            raise ValueError(f"Node {node_id} does not exist.")
    
    def get_edge_data(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Retrieve the edge data between two nodes."""
        if self.graph.has_edge(source_id, target_id):
            edge_data = self.graph[source_id][target_id]
            return {
                "type": edge_data["type"],
                "confidence": edge_data["confidence"]
            }
        else:
            raise ValueError(f"Edge between {source_id} and {target_id} does not exist.")
