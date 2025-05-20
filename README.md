# RAG Evals Repository Guide

## Overview

This repository provides simple, modular primitives for evaluating RAG (Retrieval Augmented Generation) systems using LLMs. Created by the author of Instructor, it focuses on lightweight evaluation that doesn't require extensive infrastructure.

**Repository:** [jxnl/ragas-alt](https://github.com/jxnl/ragas-alt)  
**Documentation:** [RAG Evals Documentation](https://jxnl.github.io/ragas-alt/)

## Philosophy

RAG Evals is built on the principle that RAG evaluation doesn't need to be complex. Key principles:

- **Simple Primitives**: Essential building blocks without complexity
- **Flexible Implementation**: Easily customizable prompts and evaluation logic
- **No Infrastructure Lock-in**: Runs in any environment
- **LLM Flexibility**: Works with any LLM compatible with Instructor

## Core Components

The repository is organized as:

```
rag_evals/
├── base.py              # Base evaluation classes
├── metrics/             # Metrics implementations
│   ├── __init__.py
│   ├── faithfulness.py  # Faithfulness evaluation (A|C)
│   ├── precision.py     # Context precision evaluation (C|Q)
│   └── relevance.py     # Answer relevance evaluation (A|Q)
```

## Evaluation Framework

RAG Evals implements a systematic decomposition of RAG evaluations based on the relationships between:

1. **Question (Q)**: The user's original query
2. **Context Chunks (C)**: The pieces of information retrieved from a knowledge base
3. **Answer (A)**: The response generated based on the question and context

### Implemented Metrics

1. **Context Relevance (C|Q)**: `ChunkPrecision` in `metrics/precision.py`
   - Evaluates if each context chunk is relevant to the question
   - Also known as: Context Precision, Retrieval Relevance, Contextual Relevancy

2. **Faithfulness (A|C)**: `Faithfulness` in `metrics/faithfulness.py`
   - Evaluates if the answer is factually consistent with the context
   - Also known as: Factuality, Correctness, Answer Grounding

3. **Answer Relevance (A|Q)**: `AnswerRelevance` in `metrics/relevance.py`
   - Evaluates how well the answer addresses the question
   - Also known as: Response Relevance, Query-Response Relevance

## Usage Instructions

### Installing

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import instructor
from rag_evals import Faithfulness, ChunkPrecision, AnswerRelevance

# Initialize with LLM
client = instructor.from_provider("openai/gpt-4o-mini")

# Sample RAG output
question = "What are the benefits of exercise?"
answer = "Regular exercise improves cardiovascular health and increases strength."
context = [
    "Regular physical activity improves heart health and circulation.",
    "Weight training builds muscle strength and increases bone density.",
    "The earliest Olympic games were held in Ancient Greece."
]

# Evaluate faithfulness
faithfulness_result = Faithfulness.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)

# Evaluate context precision
precision_result = ChunkPrecision.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)

# Evaluate answer relevance
relevance_result = AnswerRelevance.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)
```

### Parallel Evaluation

For better performance, use async methods:

```python
import asyncio
from instructor import AsyncInstructor

async_client = AsyncInstructor(provider="openai/gpt-4o-mini")

async def run_evals():
    faithfulness_task = Faithfulness.agrade(question, answer, context, async_client)
    precision_task = ChunkPrecision.agrade(question, answer, context, async_client)
    relevance_task = AnswerRelevance.agrade(question, answer, context, async_client)
    
    results = await asyncio.gather(
        faithfulness_task, precision_task, relevance_task
    )
    
    return results
```

### Customizing Prompts

All evaluation logic is in customizable prompts:

```python
from rag_evals import base, FaithfulnessResult

CustomFaithfulness = base.ContextEvaluation(
    prompt="Your custom prompt here...",
    response_model=FaithfulnessResult
)
```

## Documentation

Documentation is built with MkDocs using the Material theme and is organized as:

- **Home**: Overview and quick start
- **Metrics**:
  - Overview of all metrics
  - Faithfulness details
  - Context Precision details
  - Answer Relevance details
  - Systematic Decomposition of evaluations
- **Usage Guide**:
  - Basic usage
  - Customization
  - Examples
  - Best Practices
  - Troubleshooting
- **API Reference**: Detailed API docs

### Building Documentation

To build the documentation:

```bash
python -m mkdocs build
```

To serve locally:

```bash
python -m mkdocs serve
```

## Development Notes

- Use uv instead of pip for package management
- Avoid mocking in tests - test with real functionality
- Keep all prompts and models in the `metrics/` directory
- Follow the systematic decomposition approach when adding new metrics
- Ensure compatibility with Instructor for all metrics

## Future Plans

- **Additional Metrics**: Context Recall, Chunk Utility, etc.
- **Batch Processing**: Efficiently evaluate large datasets
- **Data Inspection Tools**: Visualize and explore results
- **Reference Implementations**: Common evaluation pipelines