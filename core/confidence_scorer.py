import torch.nn as nn

class ConfidenceScorer(nn.Module):
    """
    Module for scoring the confidence of memory outputs
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, memory_output: torch.Tensor) -> torch.Tensor:
        """Calculate confidence score for memory output"""
        return self.confidence_network(memory_output)