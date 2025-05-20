# Contributing to RAG Evals

Thank you for your interest in contributing to RAG Evals! This document outlines the process for contributing to the project and provides guidelines to ensure a smooth collaboration.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Style](#coding-style)
- [Documentation](#documentation)
- [Testing](#testing)

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the Repository**: Start by forking the [RAG Evals repository](https://github.com/jxnl/rag-evals).

2. **Clone Your Fork**: Clone your fork to your local machine:
   ```bash
   git clone https://github.com/YOUR-USERNAME/rag-evals.git
   cd rag-evals
   ```

3. **Set Up Remote**: Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/jxnl/rag-evals.git
   ```

## Development Environment

1. **Create a Virtual Environment**: We recommend using a virtual environment for development:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**: Install the development dependencies:
   ```bash
   # Install with uv (preferred)
   uv pip install -e ".[test,dev]"
   
   # Or with pip
   pip install -e ".[test,dev]"
   ```

3. **Install Pre-commit Hooks**: 
   ```bash
   pre-commit install
   ```

## Making Contributions

### Types of Contributions

We welcome various types of contributions, including:

1. **Bug Fixes**: If you find a bug, please create an issue first, then submit a PR with the fix.
2. **Feature Additions**: New features that align with the project's goals.
3. **Documentation Improvements**: Enhancements to documentation, examples, and tutorials.
4. **New Metrics**: Implementations of new evaluation metrics for RAG systems.
5. **Performance Improvements**: Optimizations to make the library faster or more efficient.

### Contribution Workflow

1. **Create a Branch**: Create a branch for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your changes, adhering to the coding style guidelines.

3. **Write Tests**: Add tests for your changes to ensure they work correctly.

4. **Update Documentation**: Update relevant documentation, including docstrings and examples.

5. **Commit Changes**: Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add: description of your changes"
   ```

6. **Stay Updated**: Regularly sync your fork with the upstream repository:
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

## Pull Request Process

1. **Create Pull Request**: Push your changes to your fork and create a pull request to the main repository.

2. **PR Description**: Include a detailed description of your changes, including:
   - What problem your PR solves
   - How your implementation works
   - Any design decisions you made
   - How to test your changes

3. **Review Process**: Maintainers will review your PR, potentially requesting changes.

4. **Address Feedback**: Make any requested changes and push them to your branch.

5. **Merge**: Once approved, maintainers will merge your PR.

## Coding Style

We follow these coding style guidelines:

1. **PEP 8**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.

2. **Type Hints**: Use type hints for function parameters and return values.

3. **Docstrings**: Use Google-style docstrings for all functions, classes, and methods:
   ```python
   def function(param1: str, param2: int) -> bool:
       """Short description of the function.
       
       More detailed description if needed.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ValueError: When and why this error is raised
       """
   ```

4. **Imports**: Organize imports in the following order:
   - Standard library imports
   - Related third-party imports
   - Local application/library-specific imports

5. **Line Length**: Maximum line length is 88 characters (following Black defaults).

## Documentation

Good documentation is crucial for the project's usability:

1. **Module/Class/Function Docstrings**: Include detailed docstrings for all public APIs.

2. **Examples**: Add examples for new features or complex usage patterns.

3. **MkDocs**: Update MkDocs documentation for significant changes:
   ```bash
   # Build documentation
   python -m mkdocs build
   
   # Serve locally to preview
   python -m mkdocs serve
   ```

4. **README Updates**: Update README.md if your changes affect high-level functionality.

## Testing

All contributions should include appropriate tests:

1. **Testing Framework**: We use pytest for testing.

2. **Running Tests**:
   ```bash
   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/test_specific_file.py
   
   # Run with coverage
   pytest --cov=rag_evals
   ```

3. **Test Coverage**: Aim for high test coverage for new code.

4. **Test Naming**: Name test functions clearly (e.g., `test_function_expected_behavior_when_condition`).

5. **CI Checks**: Ensure all CI checks pass on your PR.

## Feedback and Questions

If you have any questions or need help with the contribution process, please open an issue in the repository, and maintainers will assist you.

Thank you for contributing to RAG Evals!