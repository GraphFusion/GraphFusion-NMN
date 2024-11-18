import torch
from core.memory_cell import MemoryCell
from core.knowledge_graph import KnowledgeGraph
from typing import Dict, List, Optional

class GraphFusion:
    """
    GraphFusion SDK: Combines Neural Memory Network and Knowledge Graph.
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 confidence_threshold: float = 0.8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.memory_cells: Dict[str, MemoryCell] = {}
        self.knowledge_graph = KnowledgeGraph()
        self.global_memory = torch.zeros(hidden_size)
    
    def process(self, 
                input_data: torch.Tensor,
                context: Optional[Dict] = None) -> Dict:
        """
        Process input data and update the network.

        Args:
            input_data (torch.Tensor): Input tensor for processing.
            context (Dict, optional): Metadata or additional context.

        Returns:
            Dict: Output features, confidence score, and cell ID.
        """
        cell_id = self._generate_cell_id(input_data)
        
        if cell_id not in self.memory_cells:
            self.memory_cells[cell_id] = MemoryCell(
                self.input_size,
                self.hidden_size
            )
        
        output, new_memory, confidence = self.memory_cells[cell_id](
            input_data,
            self.global_memory
        )
        
        if confidence >= self.confidence_threshold:
            self.knowledge_graph.add_node(
                cell_id,
                output,
                context or {},
                confidence
            )
            self.global_memory = self._update_global_memory(
                self.global_memory,
                new_memory,
                confidence
            )
        
        return {
            'output': output.detach().numpy(),
            'confidence': confidence,
            'cell_id': cell_id
        }
    
    def query(self, 
              query_vector: torch.Tensor,
              top_k: int = 5,
              min_confidence: float = 0.0) -> List[Dict]:
        """
        Query the knowledge graph for similar nodes.

        Args:
            query_vector (torch.Tensor): Query vector to search.
            top_k (int): Number of results to return.
            min_confidence (float): Minimum confidence to consider.

        Returns:
            List[Dict]: List of matching nodes with similarity and metadata.
        """
        results = []
        
        for node_id, node_data in self.knowledge_graph.graph.nodes(data=True):
            if node_data['confidence'] >= min_confidence:
                similarity = torch.cosine_similarity(
                    query_vector.unsqueeze(0),
                    torch.tensor(node_data['features']).unsqueeze(0)
                )
                
                results.append({
                    'node_id': node_id,
                    'similarity': similarity.item(),
                    'confidence': node_data['confidence'],
                    'metadata': node_data['metadata']
                })
        
        results.sort(key=lambda x: x['similarity'] * x['confidence'], reverse=True)
        return results[:top_k]
    
    def export_graph(self, format: str = 'json') -> str:
        """
        Export the knowledge graph.

        Args:
            format (str): Export format (default is 'json').

        Returns:
            str: Serialized graph data.
        """
        return self.knowledge_graph.export(format=format)
    
    @staticmethod
    def _generate_cell_id(input_data: torch.Tensor) -> str:
        """Generate a unique cell ID."""
        return f"cell_{hash(tuple(input_data.cpu().detach().numpy().flatten()))}"
    
    @staticmethod
    def _update_global_memory(current: torch.Tensor,
                              new: torch.Tensor,
                              confidence: float,
                              decay: float = 0.99) -> torch.Tensor:
        """Update global memory with decay."""
        return current * decay + new * confidence * (1 - decay)
