# Base Classes API Reference

This document provides detailed information about the core base classes in RAG Evals.

## Core Evaluation Classes

### ContextEvaluation

`ContextEvaluation` is the base class for all context-based evaluations in RAG Evals. It provides the foundation for evaluating questions, answers, and context chunks using LLMs through the Instructor library.

```python
class ContextEvaluation(BaseModel):
    """Base class for context-based evaluations that handles common patterns
    including grading question, and optional answers against a context that is enumerated   
    with an id.
    
    This class is designed to be used as a base class for specific evaluation classes.
    It provides a common interface for evaluating questions and answers against a context.
    """
    
    prompt: str
    examples: list[Any] | None = None
    response_model: type[BaseModel]
    chunk_template: str = # Template string for structuring evaluation context
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | The system prompt for the LLM evaluation. This defines the task, criteria, and output format. |
| `examples` | `list[Any] \| None` | Optional examples to include in the evaluation for few-shot learning. |
| `response_model` | `type[BaseModel]` | A Pydantic model class defining the structure of the evaluation response. |
| `chunk_template` | `str` | Jinja2 template for structuring the evaluation context, question, answer, and chunks. |

#### Methods

##### grade

```python
def grade(
    self,
    question: str,
    answer: str | None,
    context: list[Any],
    client: Instructor,
) -> BaseModel:
    """Run an evaluation of a question and optional answer against provided context chunks."""
```

Evaluates a question and optional answer against provided context chunks.

**Parameters**:
- `question` (`str`): The question being evaluated.
- `answer` (`str | None`): The answer to evaluate. Can be `None` for evaluations that only need the question and context.
- `context` (`list[Any]`): List of context chunks to evaluate against. Each chunk will be enumerated and included in the evaluation template.
- `client` (`Instructor`): An initialized Instructor client instance used to make the evaluation request.

**Returns**:
- An instance of the `response_model` containing the structured evaluation results.

##### agrade

```python
async def agrade(
    self,
    question: str,
    answer: str | None,
    context: list[Any],
    client: AsyncInstructor,
) -> BaseModel:
    """Run an evaluation of a question and optional answer against provided context chunks asynchronously."""
```

Asynchronous version of `grade` that uses `AsyncInstructor`.

**Parameters**:
- `question` (`str`): The question being evaluated.
- `answer` (`str | None`): The answer to evaluate. Can be `None` for evaluations that only need the question and context.
- `context` (`list[Any]`): List of context chunks to evaluate against.
- `client` (`AsyncInstructor`): An initialized AsyncInstructor client instance.

**Returns**:
- An instance of the `response_model` containing the structured evaluation results.

### ContextValidationMixin

A mixin class that ensures the integrity of chunk references in RAG evaluations by validating that all chunk IDs correspond to actual context chunks. This validation is crucial for maintaining data consistency and preventing errors in evaluation metrics.

```python
class ContextValidationMixin:
    """Mixin class that ensures the integrity of chunk references in RAG evaluations by validating that all chunk IDs correspond to actual context chunks."""
    
    @field_validator('graded_chunks')
    @classmethod
    def validate_chunks_against_context(cls, chunks: list[Any], info: ValidationInfo) -> list[Any]:
        """Validate and process chunk IDs against context chunks."""
```

#### Methods

##### validate_chunks_against_context

A Pydantic field validator that ensures:
1. Raises an error if any chunk IDs don't exist in the context
2. Adds missing context chunks with a score of 0 and issues a warning
3. Maintains the original order of chunks

**Parameters**:
- `chunks` (`list[Any]`): List of chunks with IDs to validate
- `info` (`ValidationInfo`): Validation info containing context. Expected to be a dict with a key 'context' whose value is the list of actual context items.

**Returns**:
- List of valid chunks, with missing context chunks added (score=0)

**Raises**:
- `ValueError`: If any chunk IDs don't exist in the context or if the validation context in `info.context` is missing or malformed.

## Scoring Models

### ChunkScore

Represents a continuous score (0-1) for a single context chunk.

```python
class ChunkScore(BaseModel):
    id_chunk: int
    score: float = Field(ge=0.0, le=1.0, description="Score from 0-1 indicating the precision of the chunk, lower is worse")
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id_chunk` | `int` | The unique identifier for the context chunk. |
| `score` | `float` | A float between 0.0 and 1.0 representing the score of the chunk. Higher is better. |

### ChunkBinaryScore

Represents a binary (pass/fail) score for a single context chunk.

```python
class ChunkBinaryScore(BaseModel):
    id_chunk: int
    score: bool = Field(description="Whether the chunk is passed or failed")
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id_chunk` | `int` | The unique identifier for the context chunk. |
| `score` | `bool` | Boolean indicating whether the chunk passes (`True`) or fails (`False`) the evaluation criteria. |

### ChunkGraded

Container for a list of graded chunks with continuous scores. Includes validation through `ContextValidationMixin`.

```python
class ChunkGraded(BaseModel, ContextValidationMixin):
    graded_chunks: list[ChunkScore]

    @property 
    def score(self) -> float:
        return sum(chunk.score for chunk in self.graded_chunks) / len(self.graded_chunks)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graded_chunks` | `list[ChunkScore]` | List of `ChunkScore` objects, each with an ID and score. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `score` | `float` | The average score across all chunks. Calculated as the sum of all chunk scores divided by the number of chunks. |

### ChunkGradedBinary

Container for a list of graded chunks with binary scores. Includes validation through `ContextValidationMixin`.

```python
class ChunkGradedBinary(BaseModel, ContextValidationMixin):
    graded_chunks: list[ChunkBinaryScore]

    @property 
    def score(self) -> float:
        return sum(chunk.score for chunk in self.graded_chunks) / len(self.graded_chunks)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graded_chunks` | `list[ChunkBinaryScore]` | List of `ChunkBinaryScore` objects, each with an ID and boolean score. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `score` | `float` | The proportion of chunks that passed. Calculated as the sum of all boolean scores (treated as 0 or 1) divided by the number of chunks. |

## Usage Examples

Here's a simple example of how to use the base classes to create a custom evaluator:

```python
from pydantic import BaseModel, Field
from rag_evals import base

# Define your custom response model
class CustomEvaluationResult(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall evaluation score")
    detailed_notes: str = Field(description="Detailed reasoning for the score")

# Create your custom evaluator
CustomEvaluator = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator assessing the quality of a RAG system response.
    Provide an overall score from 0.0 to 1.0 and detailed notes explaining your reasoning.
    """,
    response_model=CustomEvaluationResult
)

# Use your evaluator
result = CustomEvaluator.grade(
    question="What causes climate change?",
    answer="Climate change is caused by greenhouse gas emissions.",
    context=["Greenhouse gases trap heat in the atmosphere, causing global warming."],
    client=your_instructor_client
)

# Access the results
print(f"Score: {result.overall_score}")
print(f"Notes: {result.detailed_notes}")
```

For more detailed examples and customization options, see:
- [Basic Usage](../usage/index.md)
- [Examples](../usage/examples.md)
- [Customization](../usage/customization.md)