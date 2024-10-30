def main():
    # Initialize neural memory network
    network = NeuralMemoryNetwork(
        input_size=64,
        hidden_size=128,
        confidence_threshold=0.8
    )
    
    # Example usage
    input_data = torch.randn(64)
    context = {
        'timestamp': datetime.now().isoformat(),
        'source': 'example'
    }
    
    # Process input
    result = network.process(input_data, context)
    
    # Query similar information
    query = torch.randn(64)
    similar_items = network.query(query, top_k=3)
    
    return result, similar_items

if __name__ == "__main__":
    result, similar_items = main()
    print("Processing Result:", result)
    print("Similar Items:", similar_items)