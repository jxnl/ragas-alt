# Troubleshooting RAG Evals

This guide helps you troubleshoot common issues when using RAG Evals.

## Common Issues

### LLM API Errors

**Problem**: Errors when connecting to the LLM provider.

**Possible Solutions**:
- Verify API key is valid and correctly set in your environment
- Check API rate limits and quotas
- Ensure you have the correct Instructor version for your provider
- Verify network connectivity to the LLM provider

```python
# Example of proper client initialization
import instructor
import os

# Ensure API key is set
assert os.environ.get("OPENAI_API_KEY"), "API key not found in environment"

# Initialize client with error handling
try:
    client = instructor.from_provider("openai/gpt-4o-mini")
except Exception as e:
    print(f"Failed to initialize client: {e}")
    # Handle the error appropriately
```

### Context Window Limitations

**Problem**: Evaluation fails due to exceeding the model's context window.

**Possible Solutions**:
- Reduce the size of context chunks
- Use a model with a larger context window
- Limit the amount of context provided per evaluation
- Break large evaluations into smaller batches

```python
# Example: Limiting context size
def limit_context_size(context, max_chunks=10, max_chunk_size=500):
    """Limit context to prevent exceeding context window limits"""
    # Limit number of chunks
    limited_context = context[:max_chunks]
    
    # Limit size of each chunk
    limited_context = [chunk[:max_chunk_size] for chunk in limited_context]
    
    return limited_context

# Use limited context in evaluation
faithfulness_result = Faithfulness.grade(
    question=question,
    answer=answer,
    context=limit_context_size(context),
    client=client
)
```

### Validation Errors

**Problem**: Errors related to the validation of response models.

**Possible Solutions**:
- Check if your prompt aligns with the expected response model
- Verify that chunk IDs are correctly referenced
- Ensure the LLM is producing output that matches the response model schema
- Add more explicit instructions in your prompt about the required output format

```python
# Example: Adding explicit format guidance to prompt
from rag_evals import base
from rag_evals.score_faithfulness import FaithfulnessResult

# Create custom evaluator with explicit format instructions
ExplicitFormatFaithfulness = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator assessing faithfulness.
    
    IMPORTANT: Your output MUST follow this exact JSON structure:
    {
      "statements": [
        {
          "statement": "The exact claim from the answer",
          "is_supported": true or false,
          "supporting_chunk_ids": [list of integer IDs or null]
        },
        ...
      ]
    }
    
    [Rest of prompt instructions...]
    """,
    response_model=FaithfulnessResult
)
```

### Incorrect Chunk IDs

**Problem**: The evaluation references chunk IDs that don't exist in the context.

**Possible Solutions**:
- Ensure chunk IDs start from 0 and are sequential
- Verify that the LLM understands the chunk ID format
- Check for chunk ID consistency in your prompts and examples
- Make the consequences of invalid chunk IDs explicit in your prompt

```python
# Example: Validating context chunks before evaluation
def validate_context(context):
    """Ensure context is properly formatted with sequential IDs"""
    if not context:
        raise ValueError("Context cannot be empty")
    
    # Ensure context is a list
    if not isinstance(context, list):
        raise TypeError("Context must be a list of strings")
    
    # Check that all context items are strings
    for i, chunk in enumerate(context):
        if not isinstance(chunk, str):
            raise TypeError(f"Context chunk {i} is not a string")
    
    return context

# Use validated context
faithfulness_result = Faithfulness.grade(
    question=question,
    answer=answer,
    context=validate_context(context),
    client=client
)
```

### Inconsistent Evaluation Results

**Problem**: Evaluations produce inconsistent or unexpected results.

**Possible Solutions**:
- Use a more capable LLM for evaluation
- Provide explicit scoring criteria in the prompt
- Add few-shot examples to guide the evaluation
- Run multiple evaluations and average the results
- Review your prompt for clarity and potential ambiguities

```python
# Example: Running multiple evaluations for consistency
def evaluate_with_redundancy(question, answer, context, client, evaluator, n=3):
    """Run multiple evaluations and aggregate results for more consistency"""
    results = []
    
    for _ in range(n):
        result = evaluator.grade(
            question=question,
            answer=answer,
            context=context,
            client=client
        )
        results.append(result)
    
    # For faithfulness evaluations, average overall scores
    if hasattr(results[0], 'overall_faithfulness_score'):
        avg_score = sum(r.overall_faithfulness_score for r in results) / len(results)
        print(f"Average Faithfulness Score: {avg_score:.2f}")
    
    # For precision evaluations, average across chunk scores
    elif hasattr(results[0], 'avg_score'):
        avg_score = sum(r.avg_score for r in results) / len(results)
        print(f"Average Precision Score: {avg_score:.2f}")
    
    return results
```

### Performance Issues

**Problem**: Evaluations are too slow.

**Possible Solutions**:
- Use parallel processing with `agrade` and `asyncio`
- Batch evaluations when possible
- Use a faster LLM for evaluations that don't require high capability
- Optimize context size to reduce token count
- Consider using client-side caching for repeated evaluations

```python
# Example: Parallel evaluation of multiple metrics
import asyncio
from instructor import AsyncInstructor
from rag_evals.score_faithfulness import Faithfulness
from rag_evals.score_precision import ChunkPrecision

async_client = AsyncInstructor(provider="openai/gpt-4o-mini")

async def evaluate_example(question, answer, context):
    """Run all evaluations for a single example in parallel"""
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
    
    # Run both evaluations in parallel
    return await asyncio.gather(faithfulness_task, precision_task)

# Use with asyncio.run(evaluate_example(...))
```

## Model-Specific Issues

### GPT Models

- Ensure you're using the correct model name format (e.g., "openai/gpt-4o-mini")
- Monitor token usage to avoid unexpected costs
- Be aware of rate limits, especially with parallel evaluations

### Claude Models

- Properly format the system and user messages for Claude
- Be aware of Claude's handling of structured output
- Adjust prompts to accommodate Claude's reasoning style

### Local Models

- Ensure the model supports structured JSON output
- Be prepared for less consistent results with smaller models
- Consider using more explicit prompts for local models

## Debugging Tips

1. **Inspect Raw API Responses**: Look at the raw API responses to understand what the LLM is returning
2. **Log Intermediate Steps**: Add logging to track the evaluation process
3. **Test With Simple Examples**: Verify functionality with simple, known examples first
4. **Compare With Manual Evaluation**: Periodically validate results against human judgments
5. **Check Context Processing**: Verify that context is being correctly processed and enumerated

If you continue to experience issues, please check the project repository for known issues or submit a new issue with details about your problem.