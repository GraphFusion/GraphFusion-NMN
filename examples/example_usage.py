import torch
from sdk.graphfusion import GraphFusion  # Assuming your SDK is installed as `graphfusion` package

def main():
    # Initialize the GraphFusion SDK
    fusion = GraphFusion(input_size=64, hidden_size=128)

    # Input data and context
    input_data = torch.randn(1, 64)
    context = {
        'timestamp': '2023-04-25T12:34:56',
        'source': 'user_input'
    }

    # Process the input through the network
    result = fusion.process(input_data, context)
    print("Processing Result:", result)

    # Query for similar items
    query = torch.randn(1, 64)
    similar_items = fusion.query(query, top_k=3)
    print("Similar Items:", similar_items)

    # Export the knowledge graph
    graph_data = fusion.export_graph('json')
    print("Knowledge Graph:", graph_data)

if __name__ == "__main__":
    main()
