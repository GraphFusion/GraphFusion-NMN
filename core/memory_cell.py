import torch
import torch.nn as nn
from typing import Tuple, Dict

class MemoryCell(nn.Module):
    """
    Core memory cell unit with attention and confidence mechanisms.
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Memory processing components
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.memory_processor = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        
        # Confidence scoring
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                input_data: torch.Tensor, 
                prev_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Process input and update memory."""
        # Project input to hidden size
        projected_input = self.input_projection(input_data)

        # Ensure the input shape is (batch_size, seq_len, hidden_size)
        batch_size = projected_input.size(0)  
        
        # Process through LSTM
        # Initialize hidden and cell states
        hidden = prev_memory.unsqueeze(0) if prev_memory.dim() == 2 else torch.zeros(1, batch_size, self.hidden_size).to(input_data.device)
        cell = prev_memory.unsqueeze(0) if prev_memory.dim() == 2 else torch.zeros(1, batch_size, self.hidden_size).to(input_data.device)

        memory_output, (hidden, cell) = self.memory_processor(
            projected_input.unsqueeze(1),  # Add sequence dimension: (batch_size, 1, hidden_size)
            (hidden, cell)  # Hidden and cell states with shape (num_layers, batch_size, hidden_size)
        )
        
        # Apply attention
        attended_memory, _ = self.attention(
            memory_output,
            memory_output,
            memory_output
        )
        
        # Calculate confidence
        confidence = self.confidence_scorer(attended_memory.squeeze(1))
        # Ensure the confidence tensor is a scalar before calling .item()
        if confidence.numel() == 1:  
            confidence_value = confidence.item()
        else:
            confidence_value = confidence.mean().item()

        return attended_memory.squeeze(1), hidden.squeeze(0), confidence_value

def validate_node_data(node_data: Dict) -> bool:
    """Validate node data for the knowledge graph."""
    return isinstance(node_data, dict) and 'features' in node_data

def validate_edge_data(edge_data: Dict) -> bool:
    """Validate edge data for the knowledge graph."""
    return isinstance(edge_data, dict) and 'type' in edge_data
