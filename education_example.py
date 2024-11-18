import torch
from sdk.graphfusion import GraphFusion

def education_example():
    # Initialize GraphFusion for educational use
    fusion = GraphFusion(input_size=64, hidden_size=128)

    # Simulate student performance data (e.g., test scores, completed assignments)
    input_data = torch.randn(1, 64)  # Example tensor might represent test scores, assignment completion rates, etc.
    
    # Context includes student metadata like ID, grades, and course enrolled
    context = {
        'student_id': 'student_789',
        'course': 'AI Fundamentals',
        'current_grade': 'B',
        'completed_assignments': ['assignment_1', 'assignment_2'],
        'timestamp': '2023-11-18T14:00:00'
    }

    # Process the student data
    result = fusion.process(input_data, context)
    print("Student Processing Result:", result)

    # Query for similar students who need additional support
    query = torch.randn(1, 64)  # Query could represent new student data
    similar_students = fusion.query(query, top_k=5)
    print("Similar Students for Peer Learning:", similar_students)

    # Export the knowledge graph to review student progress and interactions
    graph_data = fusion.export_graph('json')
    print("Student Knowledge Graph:", graph_data)

if __name__ == "__main__":
    education_example()
