from setuptools import setup, find_packages

# Read content for long description and release notes
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("RELEASE_NOTES.md", "r", encoding="utf-8") as fh:
    release_notes = fh.read()

setup(
    name="graphfusion",
    version="0.1.0",
    author="Your Name or Organization",
    author_email="your_email@example.com",
    description="GraphFusion: A Neural Memory Network and Knowledge Graph SDK",
    long_description=f"{long_description}\n\n## Release Notes\n\n{release_notes}",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphfusion",  # Replace with actual URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "transformers", 
        "pydantic",
    ],
    include_package_data=True,
    zip_safe=False,
)
