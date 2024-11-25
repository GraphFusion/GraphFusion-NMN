import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class MemoryCell(nn.Module):
    """
    Core memory cell unit with persistent state management and attention mechanisms.
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Core components
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.memory_processor = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        
        # Confidence scoring
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # State management
        self._hidden_state: Optional[torch.Tensor] = None
        self._cell_state: Optional[torch.Tensor] = None
        
    def _initialize_states(self, batch_size: int, device: torch.device) -> None:
        """Initialize hidden and cell states if they don't exist."""
        if self._hidden_state is None or self._cell_state is None:
            self._hidden_state = torch.zeros(1, batch_size, self.hidden_size, device=device)
            self._cell_state = torch.zeros(1, batch_size, self.hidden_size, device=device)
            
    def reset_states(self) -> None:
        """Explicitly reset the memory states when needed."""
        self._hidden_state = None
        self._cell_state = None
        
    def detach_states(self) -> None:
        """Detach states from computation graph to prevent memory leaks."""
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()
        if self._cell_state is not None:
            self._cell_state = self._cell_state.detach()
            
    def get_state(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get current memory states."""
        return self._hidden_state, self._cell_state
    
    def set_state(self, hidden: torch.Tensor, cell: torch.Tensor) -> None:
        """Set memory states explicitly."""
        self._hidden_state = hidden
        self._cell_state = cell
        
    def forward(self, 
                input_data: torch.Tensor,
                prev_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Process input and update memory with state persistence.
        
        Args:
            input_data: Input tensor of shape (batch_size, input_size)
            prev_memory: Previous memory tensor of shape (batch_size, hidden_size)
            
        Returns:
            Tuple containing:
            - Attended memory output
            - New hidden state
            - Confidence score
        """
        batch_size = input_data.size(0)
        device = input_data.device
        
        # Initialize or validate states
        self._initialize_states(batch_size, device)
        
        # Project input
        projected_input = self.input_projection(input_data)
        
        # Process through LSTM with persistent states
        memory_output, (new_hidden, new_cell) = self.memory_processor(
            projected_input.unsqueeze(1),
            (self._hidden_state, self._cell_state)
        )
        
        # Update persistent states
        self._hidden_state = new_hidden
        self._cell_state = new_cell
        
        # Apply attention mechanism
        attended_memory, _ = self.attention(
            memory_output,
            memory_output,
            memory_output
        )
        
        # Calculate confidence
        confidence = self.confidence_scorer(attended_memory.squeeze(1))
        confidence_value = confidence.mean().item()
        
        # Detach states for memory efficiency
        self.detach_states()
        
        return attended_memory.squeeze(1), new_hidden.squeeze(0), confidence_value


# Example usage:
def test_memory_cell():
    # Create test instance
    input_size = 64
    hidden_size = 128
    memory_cell = MemoryCell(input_size, hidden_size)
    
    # Create sample input
    batch_size = 2
    input_tensor = torch.randn(batch_size, input_size)
    prev_memory = torch.zeros(batch_size, hidden_size)
    
    # First forward pass
    output1, hidden1, conf1 = memory_cell(input_tensor, prev_memory)
    
    # Second forward pass (should use persistent state)
    output2, hidden2, conf2 = memory_cell(input_tensor, output1)
    
    # Reset states when needed
    memory_cell.reset_states()
    
    return {
        "output1_shape": output1.shape,
        "output2_shape": output2.shape,
        "hidden1_shape": hidden1.shape,
        "hidden2_shape": hidden2.shape,
        "conf1": conf1,
        "conf2": conf2
    }

if __name__ == "__main__":
    results = test_memory_cell()
    print(results)
