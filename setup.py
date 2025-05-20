from setuptools import setup, find_packages

setup(
    name="ragas-alt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        # e.g., "numpy>=1.0",
    ],
    # Add other metadata like author, description, etc.
    author="Jason Li",
    author_email="jason@jxnl.co",
    description="RAGAs-Alt is a library for evaluating RAG systems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jxnl/ragas-alt",
)
