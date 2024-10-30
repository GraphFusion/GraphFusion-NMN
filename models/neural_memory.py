from typing import Dict, List, Optional
import torch

class NeuralMemoryNetwork:
    """
    Main neural memory network combining memory cells and knowledge graph
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
        Process input through the neural memory network
        """
        # Generate cell ID based on input characteristics
        cell_id = self._generate_cell_id(input_data)
        
        # Get or create memory cell
        if cell_id not in self.memory_cells:
            self.memory_cells[cell_id] = MemoryCell(
                self.input_size,
                self.hidden_size
            )
        
        # Process through memory cell
        output, new_memory, confidence = self.memory_cells[cell_id](
            input_data,
            self.global_memory
        )
        
        # Update knowledge graph if confidence is high enough
        if confidence >= self.confidence_threshold:
            self.knowledge_graph.add_node(
                cell_id,
                output,
                context or {},
                confidence
            )
            
            # Update global memory
            self.global_memory = self._update_global_memory(
                self.global_memory,
                new_memory,
                confidence
            )
        
        return {
            'output': output,
            'confidence': confidence,
            'cell_id': cell_id
        }
    
    def query(self, 
              query_vector: torch.Tensor,
              top_k: int = 5,
              min_confidence: float = 0.0) -> List[Dict]:
        """
        Query the memory network for similar information
        """
        results = []
        
        # Get relevant subgraph
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
        
        # Sort by similarity * confidence
        results.sort(
            key=lambda x: x['similarity'] * x['confidence'],
            reverse=True
        )
        
        return results[:top_k]
    
    @staticmethod
    def _generate_cell_id(input_data: torch.Tensor) -> str:
        """Generate unique cell ID based on input characteristics"""
        # This is a simple implementation - customize based on your needs
        return f"cell_{hash(tuple(input_data.tolist()))}"
    
    @staticmethod
    def _update_global_memory(current: torch.Tensor,
                            new: torch.Tensor,
                            confidence: float,
                            decay: float = 0.99) -> torch.Tensor:
        """Update global memory with new information"""
        return current * decay + new * confidence * (1 - decay)