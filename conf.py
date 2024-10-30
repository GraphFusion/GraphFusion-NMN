import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

print("Project Root:", project_root)
print("sys.path:", sys.path)  # This will print out all paths where Python looks for modules
