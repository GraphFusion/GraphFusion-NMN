from sdk.graphfusion import GraphFusion
import torch

# Initialize GraphFusion
fusion = GraphFusion(input_size=128, hidden_size=256, confidence_threshold=0.8)

# Process some data
input_data = torch.randn(128)  # Random input data
context = {"source": "example_data", "description": "Randomly generated input"}

result = fusion.process(input_data, context)
print(f"Processed Result:\nOutput: {result['output']}\nConfidence: {result['confidence']}\nCell ID: {result['cell_id']}")

# Query the graph
query_vector = torch.randn(128)  # Random query vector
query_results = fusion.query(query_vector, top_k=3)

print("\nQuery Results:")
for idx, res in enumerate(query_results, start=1):
    print(f"{idx}. Node ID: {res['node_id']} | Similarity: {res['similarity']:.4f} | Confidence: {res['confidence']:.4f}")

# Export the graph
exported_graph = fusion.export_graph(format='json')
with open("knowledge_graph.json", "w") as file:
    file.write(exported_graph)
print("\nKnowledge graph has been exported to 'knowledge_graph.json'.")