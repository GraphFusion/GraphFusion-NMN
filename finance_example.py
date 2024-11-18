import torch
from sdk.graphfusion import GraphFusion

def finance_example():
    # Initialize GraphFusion for financial data analysis
    fusion = GraphFusion(input_size=64, hidden_size=128)

    # Simulate transaction data (e.g., amount, sender, receiver, time of transaction)
    input_data = torch.randn(1, 64)  # Example tensor could represent features like transaction amount, etc.
    
    # Context includes transaction metadata like sender ID, transaction type, etc.
    context = {
        'transaction_id': 'txn_98765',
        'sender_id': 'user_123',
        'receiver_id': 'user_456',
        'amount': 1500,
        'timestamp': '2023-11-18T12:15:00',
        'transaction_type': 'transfer'
    }

    # Process the transaction data
    result = fusion.process(input_data, context)
    print("Transaction Processing Result:", result)

    # Query for similar fraudulent transactions based on past patterns
    query = torch.randn(1, 64)  # Query might represent a new transaction for fraud detection
    similar_transactions = fusion.query(query, top_k=5)
    print("Similar Transactions (Potential Fraud):", similar_transactions)

    # Export the knowledge graph to review transaction relationships
    graph_data = fusion.export_graph('json')
    print("Transaction Knowledge Graph:", graph_data)

if __name__ == "__main__":
    finance_example()
