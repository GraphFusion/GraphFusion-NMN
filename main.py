import torch
from core.memory_cell import MemoryCell, validate_node_data, validate_edge_data
from core.knowledge_graph import KnowledgeGraph
from models.neural_memory import NeuralMemoryNetwork

def main():
    # Initialize components
    memory_cell = MemoryCell(input_size=64, hidden_size=128)
    knowledge_graph = KnowledgeGraph()
    neural_memory = NeuralMemoryNetwork(input_size=64, hidden_size=128)

    # Example usage
    input_data = torch.randn(1, 64)  # Change to shape (batch_size, input_size)
    context = {
        'timestamp': '2023-04-25T12:34:56',
        'source': 'user_input'
    }

    # Process input through memory cell
    output, new_memory, confidence = memory_cell(input_data, torch.zeros(1, 128))  # Ensure hidden state is the right shape

    # Add node to knowledge graph
    node_data = {
        'features': output.detach(),  # Detach to prevent gradient tracking
        'metadata': context,
        'confidence': confidence
    }

    # Validate and add the node to the knowledge graph
    if validate_node_data(node_data):
        knowledge_graph.add_node('cell_1', **node_data)

    # Add relationship to knowledge graph
    confidence = 0.8  # Use the confidence directly, without wrapping it in a dict

    # Validate and add the edge to the knowledge graph
    if validate_edge_data({'confidence': confidence}):  # Only send confidence if that's expected
        knowledge_graph.add_edge('cell_1', 'cell_2', confidence=confidence)

    # Process input through the neural memory network
    result = neural_memory.process(input_data, context)
    print("Processing Result:", result)

    # Query similar information from the neural memory network
    query = torch.randn(1, 64)  # Ensure the query has the right shape
    similar_items = neural_memory.query(query, top_k=3)
    print("Similar Items:", similar_items)

    # Export knowledge graph
    graph_data = knowledge_graph.export('json')
    print("Knowledge Graph:", graph_data)

if __name__ == "__main__":
    main()
