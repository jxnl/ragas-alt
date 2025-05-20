# Context Precision Evaluation

Context Precision (also known as Chunk Relevancy) measures whether each retrieved context chunk is relevant to the original question. This metric helps assess the efficiency of your retrieval system.

## What Context Precision Measures

- **Definition**: Context Precision evaluates for each individual **retrieved chunk** of context, how relevant its content is to the **user's original question**, regardless of whether that chunk was actually used in the final generated answer.
- **Focus**: The relationship is (Individual Retrieved Chunk → User's Question).
- **Purpose**: To determine if the retriever is fetching chunks that are relevant to the user's query. If many chunks are retrieved but aren't relevant to the question, the retrieval might be inefficient.

## How It Works

The Context Precision evaluator:

1. Examines each context chunk independently
2. Determines if the information in the chunk is relevant to answering the original question
3. Assigns a binary score (relevant/not relevant) to each chunk
4. Calculates an overall precision score based on the proportion of relevant chunks

## Implementation Details

The context precision implementation uses a straightforward approach:

```python
from rag_evals.score_precision import ChunkPrecision

precision_result = ChunkPrecision.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)
```

### Response Model

The ChunkPrecision evaluator uses the base `ChunkGradedBinary` class:

```python
class ChunkBinaryScore(BaseModel):
    id_chunk: int  # ID of the chunk being evaluated
    score: bool  # Whether the chunk is relevant (True) or not (False)
    
class ChunkGradedBinary(BaseModel, ContextValidationMixin):
    graded_chunks: list[ChunkBinaryScore]  # All evaluated chunks
    
    @property 
    def avg_score(self) -> float:
        # Calculates the proportion of relevant chunks
```

### Example Output

```python
# Result example
precision_result = ChunkGradedBinary(
    graded_chunks=[
        ChunkBinaryScore(id_chunk=0, score=True),   # Chunk is relevant to the question
        ChunkBinaryScore(id_chunk=1, score=True),   # Chunk is relevant to the question
        ChunkBinaryScore(id_chunk=2, score=False),  # Chunk is not relevant to the question
    ]
)

# Overall score: 0.6667 (2/3 chunks were relevant)
```

## Customizing the Evaluation

You can customize the context precision prompt to adjust the criteria for what makes a chunk "relevant":

```python
from rag_evals.score_precision import ChunkPrecision
from rag_evals import base

# Access the original prompt
original_prompt = ChunkPrecision.prompt

# Create a customized evaluator with a modified prompt
CustomPrecision = base.ContextEvaluation(
    prompt="Your custom prompt here...",
    response_model=base.ChunkGradedBinary
)
```

## Context Precision vs. Chunk Utility

It's important to understand the difference between:

- **Context Precision**: Measures if a chunk is relevant to the question (regardless of whether it was used in the answer)
- **Chunk Utility**: Measures if a chunk was actually used in generating the answer

A chunk might be highly relevant to the question but not used in the answer, or it might be used in the answer despite having low relevance to the question.

## Considerations When Using Context Precision

- **Partial Relevance**: Consider how to score chunks that are only partially relevant to the question
- **Topical vs. Factual Relevance**: A chunk might be topically relevant but not contain the specific facts needed
- **Question Decomposition**: For complex questions with multiple parts, chunks may be relevant to only some parts

## Best Practices

- Use context precision alongside other metrics like faithfulness for a complete evaluation
- Analyze chunks marked as "not relevant" to improve your retrieval system
- Consider both precision and recall metrics for a comprehensive view of retrieval performance
- Try different chunk sizes to find the optimal granularity for your use case

For implementation examples, see the [usage examples](../usage/examples.md).