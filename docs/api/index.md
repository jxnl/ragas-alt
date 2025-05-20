# API Reference

This section provides detailed API documentation for the RAG Evals library.

## Core Classes

### ContextEvaluation

`ContextEvaluation` is the base class for all context-based evaluations in RAG Evals.

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
    chunk_template: str = dedent("""
        <evaluation>
            {% if examples is not none %}
            <examples>
                {% for example in examples %}
                <example>
                    {{ example }}
                </example>
                {% endfor %}
            {% endif %}
            <question>{{ question }}</question>
            {% if answer is not none %}
            <answer>{{ answer }}</answer>
            {% endif %}
            <context>
                {% for chunk in context %}
                <chunk id="{{ chunk.id }}">
                    {{ chunk.chunk }}
                </chunk>
                {% endfor %}
            </context>
        </evaluation>
    """)
```

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
    """Run an evaluation of a question and optional answer against provided context chunks.

    Args:
        question (str): The question being evaluated.
        answer (Optional[str]): The answer to evaluate, if available. Can be None.
        context (List[Any]): List of context chunks to evaluate against.
        client (Instructor): An initialized Instructor client instance.

    Returns:
        BaseModel: An instance of the response_model containing the structured evaluation results.
    """
```

##### agrade

```python
async def agrade(
    self,
    question: str,
    answer: str | None,
    context: list[Any],
    client: AsyncInstructor,
) -> BaseModel:
    """Run an evaluation of a question and optional answer against provided context chunks asynchronously.

    Args:   
        question (str): The question being evaluated.   
        answer (Optional[str]): The answer to evaluate, if available. Can be None.
        context (List[Any]): List of context chunks to evaluate against.
        client (AsyncInstructor): An initialized AsyncInstructor client instance.

    Returns:
        BaseModel: An instance of the response_model containing the structured evaluation results.
    """
```

### ContextValidationMixin

A mixin class that ensures the integrity of chunk references in RAG evaluations by validating that all chunk IDs correspond to actual context chunks.

```python
class ContextValidationMixin:
    """Mixin class that ensures the integrity of chunk references in RAG evaluations
    by validating that all chunk IDs correspond to actual context chunks."""
    
    @field_validator('graded_chunks')
    @classmethod
    def validate_chunks_against_context(cls, chunks: list[Any], info: ValidationInfo) -> list[Any]:
        """Validate and process chunk IDs against context chunks."""
```

### ChunkScore

Represents a score for a single context chunk.

```python
class ChunkScore(BaseModel):
    id_chunk: int
    score: float = Field(ge=0.0, le=1.0, description="Score from 0-1 indicating the precision of the chunk, lower is worse")
```

### ChunkBinaryScore

Represents a binary (pass/fail) score for a single context chunk.

```python
class ChunkBinaryScore(BaseModel):
    id_chunk: int
    score: bool = Field(description="Whether the chunk is passed or failed")
```

### ChunkGraded

Container for a list of graded chunks with a continuous score.

```python
class ChunkGraded(BaseModel, ContextValidationMixin):
    graded_chunks: list[ChunkScore]

    @property 
    def avg_score(self) -> float:
        return sum(chunk.score for chunk in self.graded_chunks) / len(self.graded_chunks)
```

### ChunkGradedBinary

Container for a list of graded chunks with a binary score.

```python
class ChunkGradedBinary(BaseModel, ContextValidationMixin):
    graded_chunks: list[ChunkBinaryScore]

    @property 
    def avg_score(self) -> float:
        return sum(chunk.score for chunk in self.graded_chunks) / len(self.graded_chunks)
```

## Faithfulness Module

### FaithfulnessResult

The result of a faithfulness evaluation.

```python
class FaithfulnessResult(BaseModel):
    statements: list[StatementEvaluation] = Field(description="A list of all statements extracted from the answer and their evaluation.")

    @property
    def overall_faithfulness_score(self) -> float:
        if not self.statements:
            return 0.0
        supported_statements = sum(s.is_supported for s in self.statements)
        return supported_statements / len(self.statements)
```

### StatementEvaluation

Represents the evaluation of a single statement extracted from an answer.

```python
class StatementEvaluation(BaseModel):
    statement: str = Field(description="An individual claim extracted from the generated answer.")
    is_supported: bool = Field(description="Is this statement supported by the provided context chunks?")
    supporting_chunk_ids: Optional[list[int]] = Field(
        default=None, 
        description="A list of chunk IDs (0-indexed integers) from the provided context that support this statement."
    )
```

### Faithfulness

The main faithfulness evaluator.

```python
Faithfulness = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator tasked with assessing the factual faithfulness of a generated answer to its provided context...
    """,
    response_model=FaithfulnessResult
)
```

## Precision Module

### ChunkPrecision

The main context precision evaluator.

```python
ChunkPrecision = base.ContextEvaluation(
    prompt = """
    You are an expert evaluator assessing if a specific retrieved context chunk was utilized in generating a given answer...
    """, 
    response_model = base.ChunkGradedBinary
)
```

## Usage Examples

For complete usage examples, see:

- [Basic Usage](../usage/index.md)
- [Examples](../usage/examples.md)
- [Customization](../usage/customization.md)