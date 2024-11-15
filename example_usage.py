import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphfusion.sdk import GraphFusion
import torch
# Create a GraphFusion instance
fusion = GraphFusion(input_size=128, hidden_size=256, confidence_threshold=0.8)

# Example input data
input_data = torch.rand(128)  # A random input vector
context = {"source": "sensor_1", "timestamp": "2024-11-15"}

# Process input through GraphFusion
result = fusion.process_input(input_data, context)
print(result)