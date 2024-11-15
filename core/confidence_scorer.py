import torch
import torch.nn as nn

class ConfidenceScorer(nn.Module):
    """
    Module for scoring the confidence of memory outputs.
    This network predicts a confidence score (0 to 1) based on the input.
    """
    def __init__(self, hidden_size: int, dropout_rate: float = 0.5):
        """
        Initialize the ConfidenceScorer module.

        Args:
            hidden_size (int): The size of the input tensor.
            dropout_rate (float): Dropout rate to prevent overfitting (default: 0.5).
        """
        super().__init__()
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Custom weight initialization for linear layers.
        """
        for module in self.confidence_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, memory_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to calculate the confidence score for memory output.

        Args:
            memory_output (torch.Tensor): Input tensor of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Confidence scores of shape (batch_size, 1), values between 0 and 1.
        """
        return self.confidence_network(memory_output)