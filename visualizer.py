import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Generate example output tensor
output_tensor = np.array([
    -0.0211, 0.0016, 0.0095, -0.0447, 0.0196, -0.0589, -0.0082, -0.0330,
    0.0046, 0.0048, -0.0295, 0.0096, 0.0059, -0.0245, 0.0069, 0.0356,
    0.0174, -0.0392, 0.0345, -0.0485, -0.0120, -0.0052, 0.0211, 0.0793,
    0.0246, -0.0165, -0.0010, 0.0058, -0.0072, -0.0154, -0.0033, 0.0048,
    0.0012, 0.0034, -0.0263, 0.0110, 0.0365, 0.0032, 0.0044, 0.0301,
    -0.0105, 0.0206, 0.0080, 0.0058, -0.0139, -0.0188, -0.0168, -0.0259,
    -0.0080, 0.0211, 0.0209, 0.0154, -0.0646, 0.0258, 0.0105, -0.0588,
    0.0045, -0.0158, -0.0482, -0.0226, 0.0148, 0.0802, 0.0189, -0.0236,
    0.0020, -0.0162, 0.0194, 0.0174, 0.0426, -0.0388, -0.0103, -0.0404,
    -0.0018, 0.0607, 0.0182, -0.0329, -0.0092, 0.0028, -0.0121, 0.0216,
    0.0142, -0.0087, 0.0072, 0.0186, 0.0211, -0.0111, 0.0401, -0.0242,
    0.0235, 0.0229, -0.0115, 0.0150, 0.0254, -0.0442, -0.0301, -0.0025,
    -0.0099, -0.0021, -0.0090, -0.0580, 0.0267, -0.0220, -0.0004, 0.0033,
    -0.0082, 0.0182, 0.0493, -0.0293, -0.0143, 0.0228, 0.0239, 0.0136,
    0.0226, 0.0026, 0.0083, 0.0158, -0.0065, -0.0338, 0.0082, 0.0144,
    -0.0330, -0.0258, -0.0204, -0.0261, -0.0061, -0.0521, 0.0252, -0.0525
])

# 1. Output Tensor Visualization
plt.figure(figsize=(12, 6))
plt.bar(range(len(output_tensor)), output_tensor, color='b')
plt.title('Output Tensor Visualization')
plt.xlabel('Index')
plt.ylabel('Value')
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Add a line at y=0
plt.show()

# 2. Knowledge Graph Visualization
# Example knowledge graph data (nodes with features)
knowledge_graph = {
    'nodes': [{'id': 'cell_1', 'features': np.random.rand(5)}],  # Using random features for demo
    'links': []
}

# Create a directed graph
G = nx.DiGraph()

# Add nodes
for node in knowledge_graph['nodes']:
    G.add_node(node['id'], features=node['features'])

# Add links (if any)
for link in knowledge_graph['links']:
    G.add_edge(link['source'], link['target'])

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue')
node_labels = {node: G.nodes[node]['features'] for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels)
plt.title('Knowledge Graph Visualization')
plt.show()

# 3. Confidence Scores Visualization
# Example confidence scores
confidence_scores = [0.525465190410614]  # For demo purposes

plt.figure(figsize=(8, 4))
plt.bar(range(len(confidence_scores)), confidence_scores, color='g')
plt.title('Confidence Score Visualization')
plt.xlabel('Node Index')
plt.ylabel('Confidence Score')
plt.ylim(0, 1)  # Confidence score range
plt.axhline(0.5, color='gray', linewidth=0.8, linestyle='--')  # Add a line at 0.5
plt.show()
