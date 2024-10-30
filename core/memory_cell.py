import torch
import torch.nn as nn
from typing import Tuple, Dict

class MemoryCell(nn.Module):
    """
    Core memory cell unit with attention and confidence mechanisms
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
        """Process input and update memory"""
        # Project input to hidden size
        projected_input = self.input_projection(input_data)
        
        # Process through LSTM
        memory_output, (hidden, cell) = self.memory_processor(
            projected_input.unsqueeze(1),
            (prev_memory.unsqueeze(0), prev_memory.unsqueeze(0))
        )
        
        # Apply attention
        attended_memory, _ = self.attention(
            memory_output,
            memory_output,
            memory_output
        )
        
        # Calculate confidence
        confidence = self.confidence_scorer(attended_memory.squeeze(1))
        
        return attended_memory.squeeze(1), hidden.squeeze(0), confidence.item()