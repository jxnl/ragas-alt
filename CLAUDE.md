# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## RAG Evals Repository

This repository provides simple, modular primitives for evaluating RAG (Retrieval Augmented Generation) systems using LLMs. It focuses on lightweight evaluation that doesn't require extensive infrastructure.

## Code Architecture

### Core Components

The repository is structured around these key components:

1. **Base Evaluation Classes** (`rag_evals/base.py`)
   - Defines core evaluation infrastructure
   - `ContextEvaluation`: Base class for evaluations with context chunks
   - Validation mixins and models for context chunking

2. **Metric Implementations** (`rag_evals/metrics/`)
   - Each metric evaluates a specific relationship in the RAG triad (Question, Context, Answer)
   - Faithfulness (`faithfulness.py`): Evaluates if answers are factually consistent with context (A|C)
   - Precision (`precision.py`): Evaluates if context chunks are relevant to the question (C|Q)
   - Relevance (`relevance.py`): Evaluates how well answers address questions (A|Q)

3. **Evaluation Framework**
   - Systematic decomposition of RAG evaluation based on:
     - Question (Q): User's original query
     - Context Chunks (C): Retrieved information pieces
     - Answer (A): Response generated from question and context

## Development Commands

### Environment Setup

```bash
# Install dependencies with uv (preferred over pip)
uv pip install -r requirements.txt

# Install in development mode
uv pip install -e .

# Install with test dependencies
uv pip install -e ".[test]"
```

### Testing

```bash
# Run all tests with coverage
pytest

# Run a specific test file
pytest tests/test_base.py

# Run a specific test
pytest tests/test_base.py::test_chunk_graded_validation
```

### Documentation

```bash
# Build documentation
python -m mkdocs build

# Serve documentation locally
python -m mkdocs serve
```

## Testing Approach

- Avoid mocking in tests - test with real functionality
- Tests use the actual OpenAI API (requiring API key in environment)
- Test fixtures provide sample questions, answers, and context chunks

## Development Guidelines

1. **Metrics Implementation**
   - Keep all prompts and models in the `metrics/` directory
   - Follow the systematic decomposition approach when adding new metrics
   - Ensure compatibility with Instructor for all metrics

2. **Package Management**
   - Use `uv` instead of `pip` for package management 

3. **Documentation**
   - Document new metrics in both code and MkDocs
   - Follow the established documentation structure in `docs/`

4. **Testing**
   - Write comprehensive tests for all new metrics
   - Maintain test fixtures in `conftest.py`
   - Use real functionality instead of mocks