# Context Recall Evaluation

Context Recall is a critical metric for RAG evaluation. It measures whether the generated answer includes all the relevant information from the retrieved context.

## What Context Recall Measures

- **Definition**: Context Recall measures if the **generated answer** includes all the relevant information from the **retrieved context**. 
- **Focus**: The relationship is (Retrieved Context → Generated Answer).
- **Purpose**: To identify if important information from relevant context chunks is missing from the answer.

## How It Works

The Context Recall evaluator:

1. Examines each context chunk to determine if it's relevant to the question
2. For relevant chunks, assesses whether their key information is reflected in the answer
3. For relevant chunks whose information is missing, identifies what specific information is missing
4. Calculates an overall recall score based on the proportion of relevant chunks whose information is included

## Implementation Details

The Context Recall implementation follows a structured approach:

```python
from rag_evals import ContextRecall

recall_result = await ContextRecall.agrade(
    question=question,
    answer=answer,
    context=context,
    client=client
)
```

### Response Model

The `ContextRecallResult` class contains:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ChunkRecallEvaluation(BaseModel):
    id_chunk: int = Field(description="The ID of the context chunk being evaluated.")
    is_relevant: bool = Field(description="Whether this chunk contains information relevant to the question.")
    is_included: bool = Field(description="Whether information from this chunk is reflected in the answer.")
    missing_info: Optional[str] = Field(
        None, 
        description="If is_relevant=True but is_included=False, this field describes the key information that is missing from the answer."
    )

class ContextRecallResult(BaseModel):
    evaluated_chunks: List[ChunkRecallEvaluation] = Field(
        description="Evaluation of each context chunk for recall assessment."
    )
    
    @property
    def score(self) -> float:
        """
        Calculate the recall score as the proportion of relevant chunks that are included in the answer.
        """
        relevant_chunks = [chunk for chunk in self.evaluated_chunks if chunk.is_relevant]
        if not relevant_chunks:
            return 1.0  # If no chunks are relevant, recall is perfect by definition
        
        included_relevant = sum(chunk.is_included for chunk in relevant_chunks)
        return included_relevant / len(relevant_chunks)
    
    @property
    def missing_information(self) -> List[dict]:
        """
        Returns a list of missing information from relevant chunks.
        """
        return [
            {"chunk_id": chunk.id_chunk, "missing_info": chunk.missing_info}
            for chunk in self.evaluated_chunks
            if chunk.is_relevant and not chunk.is_included and chunk.missing_info
        ]
```

### Example Output

```python
# Result example
recall_result = ContextRecallResult(
    evaluated_chunks=[
        ChunkRecallEvaluation(
            id_chunk=0,
            is_relevant=True,
            is_included=True,
            missing_info=None
        ),
        ChunkRecallEvaluation(
            id_chunk=1,
            is_relevant=True,
            is_included=False,
            missing_info="Information about saline nasal sprays for congestion relief"
        ),
        ChunkRecallEvaluation(
            id_chunk=2,
            is_relevant=False,
            is_included=True,
            missing_info=None
        )
    ]
)

# Recall score: 0.5 (calculated via recall_result.score)
# Missing information can be accessed via recall_result.missing_information
```

## Customizing the Evaluation

You can customize the recall prompt to adjust the evaluation criteria:

```python
from rag_evals import ContextRecall
from rag_evals import base
from rag_evals import ContextRecallResult

# Access the original prompt
original_prompt = ContextRecall.prompt

# Create a customized evaluator with a modified prompt
CustomRecall = base.ContextEvaluation(
    prompt="Your custom prompt here...",
    response_model=ContextRecallResult
)
```

## Considerations When Using Context Recall

- **Relevance Threshold**: Consider how strictly the evaluator should determine if a chunk is relevant.
- **Information Representation**: The evaluator checks if key information is represented in the answer, not if exact wording is used.
- **Completeness vs. Conciseness**: There may be a trade-off between including all relevant information and maintaining a concise answer.

## Best Practices

- Use Context Recall alongside other metrics like Faithfulness and Answer Relevance for a complete evaluation
- Review the missing information details to understand what specific information was omitted
- Consider that some information may be intentionally omitted for brevity or relevance
- Use the `missing_information` property to get a list of all missing key information

For implementation examples, see the [usage examples](../usage/examples.md).