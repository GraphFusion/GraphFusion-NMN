import torch
from core.memory_cell import MemoryCell, validate_node_data, validate_edge_data
from core.knowledge_graph import KnowledgeGraph
from models.neural_memory import NeuralMemoryNetwork

def main():
    # Initialize components
    memory_cell = MemoryCell(input_size=64, hidden_size=128)
    knowledge_graph = KnowledgeGraph()
    neural_memory = NeuralMemoryNetwork(input_size=64, hidden_size=128)

    input_data = torch.randn(1, 64)
    context = {
        'timestamp': '2023-04-25T12:34:56',
        'source': 'user_input'
    }

    output, new_memory, confidence = memory_cell(input_data, torch.zeros(1, 128))

    # Use unpacked parameters instead of dictionary for node data
    if confidence >= 0.8:
        knowledge_graph.add_node('cell_1', output, context, confidence)

    confidence_edge = 0.8
    if confidence_edge >= 0.8:
        knowledge_graph.add_edge('cell_1', 'cell_2', relationship_type='related', confidence=confidence_edge)

    result = neural_memory.process(input_data, context)
    print("Processing Result:", result)

    query = torch.randn(1, 64)
    similar_items = neural_memory.query(query, top_k=3)
    print("Similar Items:", similar_items)

    graph_data = knowledge_graph.export('json')
    print("Knowledge Graph:", graph_data)

if __name__ == "__main__":
    main()