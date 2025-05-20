# RAG Evals: Retrieval Augmented Generation

Provides simple, modular primitives for evaluating RAG systems using LLMs.

## Philosophy

From the creator of [Instructor](https://github.com/jxnl/instructor): RAG Evals is built on the principle that RAG evaluation doesn't need to be complex. While many RAG evaluation frameworks require extensive infrastructure and setup, this project aims to provide:

- **Simple Primitives**: Just the essential building blocks you need
- **Flexible Implementation**: Easily customizable prompts and evaluation logic
- **No Infrastructure Lock-in**: Run evaluations in your existing environment
- **LLM Flexibility**: Use any LLM that works with Instructor

Similar to [Instructor](https://github.com/jxnl/instructor), RAG Evals focuses on providing the fundamental tools that companies need without imposing unnecessary complexity or infrastructure requirements.

## Overview

RAG Evals helps you evaluate the quality of your RAG (Retrieval Augmented Generation) systems across several crucial dimensions:

- How truthful are your answers? (Faithfulness)
- Are your retrieved chunks relevant to the question? (Context Precision)

### Key Features

- **LLM-Powered Evaluation**: Uses language models to perform nuanced assessments
- **Structured Output**: Leverages [Instructor](https://github.com/jxnl/instructor) and Pydantic for type-safe results
- **Modifiable Prompts**: All evaluation logic is in prompts you can easily customize
- **Parallel Execution**: Run evaluations concurrently for better performance

## Quick Start

```python
import instructor
from rag_evals.score_faithfulness import Faithfulness
from rag_evals.score_precision import ChunkPrecision

# Initialize with your preferred LLM
client = instructor.from_provider("openai/gpt-4o-mini")

# Evaluate Faithfulness
question = "What are the benefits of exercise?"
answer = "Regular exercise improves cardiovascular health and increases strength."
context = [
    "Regular physical activity improves heart health and circulation.",
    "Weight training builds muscle strength and increases bone density.",
    "The earliest Olympic games were held in Ancient Greece."
]

# Run evaluations
faithfulness_result = Faithfulness.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)

print(f"Overall Faithfulness Score: {faithfulness_result.overall_faithfulness_score}")
for statement in faithfulness_result.statements:
    print(f"- {statement.statement}: {'Supported' if statement.is_supported else 'Unsupported'}")
    if statement.is_supported:
        print(f"  Supported by chunks: {statement.supporting_chunk_ids}")

# Evaluate Context Precision
precision_result = ChunkPrecision.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)

print(f"Overall Precision Score: {precision_result.avg_score}")
for chunk in precision_result.graded_chunks:
    print(f"- Chunk {chunk.id_chunk}: {'Relevant' if chunk.score else 'Not Relevant'}")
```

For more information, check out the [metrics documentation](metrics/index.md) and the [usage guide](usage/index.md).