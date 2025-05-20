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
- Are your retrieved chunks actually used? (Context Precision)

### Key Features

- **LLM-Powered Evaluation**: Uses language models to perform nuanced assessments
- **Structured Output**: Leverages [Instructor](https://github.com/jxnl/instructor) and Pydantic for type-safe results
- **Modifiable Prompts**: All evaluation logic is in prompts you can easily customize
- **Parallel Execution**: Run evaluations concurrently for better performance

## Implemented Metrics

RAG Evals currently implements these key evaluation metrics:

### 1. Faithfulness (Answer Grounding)

Measures whether the generated answer contains only factual claims supported by the retrieved context. It helps detect hallucinations (made-up information).

- **Input**: Question, Answer, Retrieved Context
- **Output**: Statement-level breakdown of which claims are supported, with specific attribution to context chunks

### 2. Context Precision (Chunk Utility)

Evaluates whether each individual retrieved context chunk was useful in generating the answer.

- **Input**: Question, Answer, Retrieved Context
- **Output**: Binary score for each chunk indicating whether it contributed to the answer

## Installation

```bash
pip install -r requirements.txt
```

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
    print(f"- Chunk {chunk.id_chunk}: {'Used' if chunk.score else 'Not Used'}")

# Parallel execution example
import asyncio
from instructor import AsyncInstructor

async_client = AsyncInstructor(...)

async def run_evals():
    # Run multiple evaluations in parallel
    faithfulness_task = Faithfulness.agrade(
        question=question,
        answer=answer,
        context=context,
        client=async_client
    )
    
    precision_task = ChunkPrecision.agrade(
        question=question,
        answer=answer,
        context=context,
        client=async_client
    )
    
    # Await results
    faithfulness_result, precision_result = await asyncio.gather(
        faithfulness_task, 
        precision_task
    )
    
    return faithfulness_result, precision_result

# Run the async function
faithfulness_result, precision_result = asyncio.run(run_evals())
```

## Customizing Prompts

All evaluation logic is defined in prompts that you can easily modify:

```python
# Modify the faithfulness prompt
from rag_evals.score_faithfulness import Faithfulness

# Access the original prompt
original_prompt = Faithfulness.prompt

# Create a customized evaluator with your own prompt
CustomFaithfulness = base.ContextEvaluation(
    prompt="Your custom prompt here...",
    response_model=FaithfulnessResult
)
```

## Documentation

For more detailed information about RAGAS evaluation metrics and best practices:

- [Understanding RAG Evaluation Metrics](rag-evals/metrics_explained.md)
- [RAGAS Evaluation Best Practices](rag-evals/tips.md)

## Future Work

Future plans include:

- **Data Inspection Tools**: Utilities to explore and visualize evaluation results
- **Additional Metrics**: Answer Relevancy, Context Recall, etc.
- **Batch Processing**: Efficiently evaluate large datasets
- **Reference Implementations**: Common evaluation pipelines and patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
