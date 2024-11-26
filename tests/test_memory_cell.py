import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from core.memory_cell import MemoryCell  

def test_memory_cell():
    """
    Test the basic functionality of the MemoryCell.
    """
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
    
    print("Test Memory Cell Results:")
    print({
        "output1_shape": output1.shape,
        "output2_shape": output2.shape,
        "hidden1_shape": hidden1.shape,
        "hidden2_shape": hidden2.shape,
        "conf1": conf1,
        "conf2": conf2
    })

def test_state_persistence():
    """
    Test state persistence and manual reset functionality in the MemoryCell.
    """
    input_size, hidden_size = 64, 128
    memory_cell = MemoryCell(input_size, hidden_size)

    # Initial forward pass
    batch_size = 2
    input_tensor1 = torch.randn(batch_size, input_size)
    prev_memory1 = torch.zeros(batch_size, hidden_size)
    
    # First forward pass
    output1, hidden1, _ = memory_cell(input_tensor1, prev_memory1)

    # Capture the current hidden state after first pass
    stored_hidden_state1 = memory_cell._hidden_state.squeeze(0)

    # Second forward pass with the same input
    input_tensor2 = input_tensor1.clone()
    output2, hidden2, _ = memory_cell(input_tensor2, output1)

    # Capture the updated hidden state
    stored_hidden_state2 = memory_cell._hidden_state.squeeze(0)

    # Assertions with more flexible tolerance
    assert torch.allclose(hidden1, stored_hidden_state1, atol=1e-4), "First hidden state not persisted correctly"
    assert torch.allclose(hidden2, stored_hidden_state2, atol=1e-4), "Second hidden state not updated correctly"
    
    # Verify that the hidden states are different after two passes
    assert not torch.allclose(stored_hidden_state1, stored_hidden_state2, atol=1e-4), "Hidden states should evolve"

    # Reset states and test
    memory_cell.reset_states()
    assert memory_cell._hidden_state is None and memory_cell._cell_state is None, "States not reset"

if __name__ == "__main__":
    print("Running test_memory_cell...")
    test_memory_cell()
    print("\nRunning test_state_persistence...")
    test_state_persistence()
