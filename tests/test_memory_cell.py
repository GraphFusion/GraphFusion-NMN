import torch
from core.memory_cell import MemoryCell

def test_memory_cell_state_persistence():
    # Setup
    input_size, hidden_size = 64, 128
    memory_cell = MemoryCell(input_size, hidden_size)
    batch_size = 2
    input_tensor = torch.randn(batch_size, input_size)
    
    # First forward pass
    _, hidden1, _ = memory_cell(input_tensor, torch.zeros(batch_size, hidden_size))
    
    # Second forward pass
    _, hidden2, _ = memory_cell(input_tensor, torch.zeros(batch_size, hidden_size))
    
    # Assert states persist
    assert torch.equal(hidden1, memory_cell._hidden_state.squeeze(0)), "Hidden state not persisted"
    
    # Reset states and check
    memory_cell.reset_states()
    assert memory_cell._hidden_state is None, "States not reset properly"
