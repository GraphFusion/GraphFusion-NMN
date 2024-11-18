import torch
from sdk.graphfusion import GraphFusion

def healthcare_example():
    # Initialize GraphFusion for healthcare
    fusion = GraphFusion(input_size=64, hidden_size=128)

    # Simulate patient data (e.g., medical records, symptoms, lab results)
    input_data = torch.randn(1, 64)  # Example tensor, could represent features like age, blood pressure, etc.
    
    # Context includes patient metadata like ID, age, etc.
    context = {
        'patient_id': '12345',
        'age': 45,
        'gender': 'M',
        'medical_history': ['hypertension', 'diabetes'],
        'timestamp': '2023-11-18T10:00:00'
    }

    # Process the patient data
    result = fusion.process(input_data, context)
    print("Patient Processing Result:", result)

    # Query for similar patients based on medical history
    query = torch.randn(1, 64)  # Query might represent a new patient with some medical data
    similar_patients = fusion.query(query, top_k=5)
    print("Similar Patients Based on Medical History:", similar_patients)

    # Export the knowledge graph to review tracked patient interactions
    graph_data = fusion.export_graph('json')
    print("Patient Knowledge Graph:", graph_data)

if __name__ == "__main__":
    healthcare_example()
