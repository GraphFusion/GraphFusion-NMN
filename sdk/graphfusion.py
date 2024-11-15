from core.memory_cell import MemoryCell
from core.knowledge_graph import KnowledgeGraph
from models.neural_memory import NeuralMemoryNetwork
import torch
from typing import Dict, List, Optional


class GraphFusion:
    """
    High-level API for GraphFusion, enabling easy access to the neural memory network
    and knowledge graph functionalities.
    """
    def __init__(self, input_size: int, hidden_size: int, confidence_threshold: float = 0.8):
        """
        Initialize GraphFusion with neural memory network and knowledge graph.

        Args:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Size of the hidden layer in the memory network.
            confidence_threshold (float): Minimum confidence score to update the knowledge graph.
        """
        self.network = NeuralMemoryNetwork(input_size, hidden_size, confidence_threshold)

    def process_input(self, input_data: torch.Tensor, context: Optional[Dict] = None) -> Dict:
        """
        Process input through the neural memory network.

        Args:
            input_data (torch.Tensor): Input tensor to process.
            context (Optional[Dict]): Additional metadata or context.

        Returns:
            Dict: Output from the memory cell, including processed output, confidence, and cell ID.
        """
        return self.network.process(input_data, context)

    def query(self, query_vector: torch.Tensor, top_k: int = 5, min_confidence: float = 0.0) -> List[Dict]:
        """
        Query the knowledge graph for information similar to the query vector.

        Args:
            query_vector (torch.Tensor): Query vector to match against stored knowledge.
            top_k (int): Number of top results to return.
            min_confidence (float): Minimum confidence for nodes to be considered.

        Returns:
            List[Dict]: List of matched nodes, each containing similarity, confidence, and metadata.
        """
        return self.network.query(query_vector, top_k, min_confidence)

    def get_graph(self) -> KnowledgeGraph:
        """
        Access the underlying knowledge graph.

        Returns:
            KnowledgeGraph: The knowledge graph object for direct graph manipulations.
        """
        return self.network.knowledge_graph

    def add_custom_node(self, 
                        node_id: str, 
                        features: torch.Tensor, 
                        metadata: Dict, 
                        confidence: float) -> None:
        """
        Add a custom node to the knowledge graph.

        Args:
            node_id (str): Unique ID for the node.
            features (torch.Tensor): Feature vector for the node.
            metadata (Dict): Metadata dictionary.
            confidence (float): Confidence score for the node.
        """
        self.network.knowledge_graph.add_node(node_id, features, metadata, confidence)

    def add_custom_edge(self, 
                        source_id: str, 
                        target_id: str, 
                        relationship_type: str, 
                        confidence: float) -> None:
        """
        Add a custom edge to the knowledge graph.

        Args:
            source_id (str): ID of the source node.
            target_id (str): ID of the target node.
            relationship_type (str): Type of the relationship.
            confidence (float): Confidence score for the edge.
        """
        self.network.knowledge_graph.add_edge(source_id, target_id, relationship_type, confidence)
