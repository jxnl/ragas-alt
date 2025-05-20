# RAG Evals Usage Guide

This guide covers how to use RAG Evals to evaluate your RAG (Retrieval Augmented Generation) systems.

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

Here's how to perform a basic evaluation using RAG Evals:

```python
import instructor
from rag_evals.score_faithfulness import Faithfulness
from rag_evals.score_precision import ChunkPrecision

# Initialize with your preferred LLM
client = instructor.from_provider("openai/gpt-4o-mini")

# Define your evaluation inputs
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

precision_result = ChunkPrecision.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)
```

## Parallel Execution

For better performance, you can run evaluations in parallel:

```python
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
from rag_evals.score_faithfulness import Faithfulness, FaithfulnessResult
from rag_evals import base

# Access the original prompt
original_prompt = Faithfulness.prompt

# Create a customized evaluator with your own prompt
CustomFaithfulness = base.ContextEvaluation(
    prompt="Your custom prompt here...",
    response_model=FaithfulnessResult
)
```

## Working with Different LLMs

RAG Evals is designed to work with any LLM that's compatible with Instructor:

```python
# Using OpenAI
openai_client = instructor.from_provider("openai/gpt-4-turbo")

# Using Anthropic
anthropic_client = instructor.from_provider("anthropic/claude-3-opus")

# Using a local model
local_client = instructor.from_provider("local/mistral-large")
```

## Advanced Usage

For more advanced usage, check out:

- [Customization Guide](customization.md) - Learn how to customize the evaluation logic
- [Best Practices](best_practices.md) - Tips for effective RAG evaluation
- [Examples](examples.md) - Real-world examples of RAG Evals in action
- [API Reference](../api/index.md) - Detailed API documentation

## Troubleshooting

If you encounter issues:

1. Ensure your LLM has sufficient context window for the evaluation
2. Check that your prompts are clear and provide sufficient guidance
3. Verify that your context chunks are properly formatted
4. Ensure you're using a compatible version of Instructor

For more help, see the [troubleshooting guide](troubleshooting.md) or open an issue on GitHub.