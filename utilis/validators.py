from typing import Any, Dict

def validate_node_data(node_data: Dict[str, Any]) -> bool:
    """
    Validate node data to be added to the knowledge graph
    """
    required_fields = ['features', 'metadata', 'confidence']
    for field in required_fields:
        if field not in node_data:
            return False
    
    # Add additional validation rules as needed
    return True

def validate_edge_data(edge_data: Dict[str, Any]) -> bool:
    """
    Validate edge data to be added to the knowledge graph
    """
    required_fields = ['type', 'confidence']
    for field in required_fields:
        if field not in edge_data:
            return False
    
    # Add additional validation rules as needed
    return True

