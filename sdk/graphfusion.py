import torch
from core.memory_cell import MemoryCell
from core.knowledge_graph import KnowledgeGraph
from models.neural_memory import NeuralMemoryNetwork

class GraphFusion:
    def __init__(self, input_size: int, hidden_size: int, confidence_threshold: float = 0.8):
        # Initialize components
        self.memory_cell = MemoryCell(input_size=input_size, hidden_size=hidden_size)
        self.knowledge_graph = KnowledgeGraph()
        self.neural_memory = NeuralMemoryNetwork(input_size=input_size, hidden_size=hidden_size, confidence_threshold=confidence_threshold)
        self.global_memory = torch.zeros(hidden_size)  # Global memory state

    def process(self, input_data: torch.Tensor, context: dict = None):
        """Process input through the memory network."""
        # Process through MemoryCell
        output, new_memory, confidence = self.memory_cell(input_data, self.global_memory)

        # Add to knowledge graph if confidence is above threshold
        if confidence >= 0.8:
            self.knowledge_graph.add_node('cell_1', output, context or {}, confidence)

        # Update global memory
        self.global_memory = self._update_global_memory(self.global_memory, new_memory, confidence)

        # Process through NeuralMemoryNetwork
        result = self.neural_memory.process(input_data, context)
        return result

    def query(self, query_vector: torch.Tensor, top_k: int = 5):
        """Query the neural memory for similar nodes."""
        return self.neural_memory.query(query_vector, top_k)

    def _update_global_memory(self, current: torch.Tensor, new: torch.Tensor, confidence: float, decay: float = 0.99):
        """Update global memory state with new information."""
        return current * decay + new * confidence * (1 - decay)

    def export_graph(self, format: str = 'json'):
        """Export the knowledge graph in the specified format."""
        return self.knowledge_graph.export(format)
