# API Reference

This section provides detailed API documentation for the RAG Evals library.

## Module Overview

The RAG Evals library consists of these key modules:

- **Base Module**: Core classes for evaluation infrastructure
- **Metrics Module**: Implementations of specific evaluation metrics
- **Utility Module**: Helper functions for processing evaluation results

## Base Module

The base module provides the foundational classes for all RAG evaluations. For detailed documentation of all base classes, see:

- [Base Classes API Reference](./base.md)

### Core Classes Summary

- `ContextEvaluation`: Base class for all context-based evaluations
- `ContextValidationMixin`: Validates chunk references against context
- `ChunkScore`: Represents a continuous score for a chunk
- `ChunkBinaryScore`: Represents a binary score (pass/fail) for a chunk
- `ChunkGraded`: Container for graded chunks with continuous scores
- `ChunkGradedBinary`: Container for graded chunks with binary scores

## Metrics Module

The metrics module provides implementations of specific RAG evaluation metrics:

### Faithfulness

Evaluates whether statements in the answer are supported by the context.

```python
class FaithfulnessResult(BaseModel):
    statements: list[StatementEvaluation] = Field(description="A list of all statements extracted from the answer and their evaluation.")

    @property
    def score(self) -> float:
        if not self.statements:
            return 0.0
        supported_statements = sum(s.is_supported for s in self.statements)
        return supported_statements / len(self.statements)
```

```python
class StatementEvaluation(BaseModel):
    statement: str = Field(description="An individual claim extracted from the generated answer.")
    is_supported: bool = Field(description="Is this statement supported by the provided context chunks?")
    supporting_chunk_ids: Optional[list[int]] = Field(
        default=None, 
        description="A list of chunk IDs (0-indexed integers) from the provided context that support this statement."
    )
```

```python
Faithfulness = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator tasked with assessing the factual faithfulness of a generated answer to its provided context...
    """,
    response_model=FaithfulnessResult
)
```

### Precision

Evaluates whether each context chunk is relevant to the question.

```python
ChunkPrecision = base.ContextEvaluation(
    prompt = """
    You are an expert evaluator assessing if a specific retrieved context chunk is relevant to the original question...
    """, 
    response_model = base.ChunkGradedBinary
)
```

### Relevance

Evaluates how well the answer addresses the original question.

```python
class RelevanceScore(BaseModel):
    """Represents the evaluation of answer relevance to the question."""
    overall_score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Score from 0-1 indicating the relevance of the answer to the question, higher is better"
    )
    topical_match: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Score from 0-1 indicating how well the answer's topic matches the question"
    )
    completeness: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Score from 0-1 indicating how completely the answer addresses all aspects of the question"
    )
    conciseness: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Score from 0-1 indicating how concise and focused the answer is without irrelevant information"
    )
    reasoning: str = Field(
        description="Explanation of the reasoning behind the scores"
    )
```

```python
AnswerRelevance = base.ContextEvaluation(
    prompt = """
    You are an expert evaluator assessing how well an answer addresses the original question...
    """,
    response_model = RelevanceScore
)
```

## Usage Examples

For complete usage examples, see:

- [Basic Usage](../usage/index.md)
- [Examples](../usage/examples.md)
- [Customization](../usage/customization.md)