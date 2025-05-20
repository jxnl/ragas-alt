# Faithfulness Evaluation

Faithfulness is one of the most critical metrics for RAG evaluation. It measures whether the generated answer contains only factual claims that are supported by the retrieved context.

## What Faithfulness Measures

- **Definition**: Faithfulness measures if the system's **generated answer** is factually supported by the **retrieved context**. 
- **Focus**: The relationship is (Generated Answer → Retrieved Context).
- **Purpose**: To detect hallucinations (made-up information) or contradictions between the answer and the context.

## How It Works

The Faithfulness evaluator:

1. Breaks down the answer into individual, verifiable statements
2. For each statement, determines if it is supported by the provided context
3. Identifies which specific context chunks support each statement
4. Calculates an overall faithfulness score based on the proportion of supported statements

## Implementation Details

The faithfulness implementation uses a structured approach:

```python
from rag_evals.score_faithfulness import Faithfulness

faithfulness_result = Faithfulness.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)
```

### Response Model

The `FaithfulnessResult` class contains:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class StatementEvaluation(BaseModel):
    statement: str = Field(description="An individual claim extracted from the generated answer.")
    is_supported: bool = Field(description="Is this statement supported by the provided context chunks?")
    supporting_chunk_ids: Optional[List[int]] = Field(
        default=None,
        description="A list of chunk IDs (0-indexed integers) from the provided context that support this statement. Null if not supported or if IDs are not applicable/found."
    )
    # Note: The evaluation prompt for Faithfulness (used by the LLM to generate the data for this model)
    # also requests a 'reasoning' field for each statement. However, this 'reasoning' field
    # is not currently defined in the StatementEvaluation Pydantic model above. Therefore,
    # while the LLM might generate reasoning, it won't be parsed into and available in the
    # FaithfulnessResult object.

class FaithfulnessResult(BaseModel):
    statements: List[StatementEvaluation] = Field(description="A list of all statements extracted from the answer and their evaluation.")

    @property
    def score(self) -> float:
        """
        Calculates the overall faithfulness score.
        This is the proportion of statements that are supported by the context.
        """
        if not self.statements:
            return 0.0
        supported_statements = sum(s.is_supported for s in self.statements)
        return supported_statements / len(self.statements)
```

### Example Output

```python
# Result example
faithfulness_result = FaithfulnessResult(
    statements=[
        StatementEvaluation(
            statement="Regular exercise improves cardiovascular health", 
            is_supported=True, 
            supporting_chunk_ids=[0]
        ),
        StatementEvaluation(
            statement="Exercise increases strength", 
            is_supported=True, 
            supporting_chunk_ids=[1]
        )
    ]
)

# Faithfulness score: 1.0 (calculated via faithfulness_result.score)
```

## Customizing the Evaluation

You can customize the faithfulness prompt to adjust the evaluation criteria:

```python
from rag_evals.score_faithfulness import Faithfulness
from rag_evals import base

# Access the original prompt
original_prompt = Faithfulness.prompt

# Create a customized evaluator with a modified prompt
from rag_evals.score_faithfulness import FaithfulnessResult

CustomFaithfulness = base.ContextEvaluation(
    prompt="Your custom prompt here...",
    response_model=FaithfulnessResult
)
```

## Considerations When Using Faithfulness

- **Strict vs. Lenient Evaluation**: Consider how strict you want the evaluation to be. Should inferences and logical conclusions be considered faithful if they're not explicitly stated in the context?
- **Granularity of Statements**: The way statements are broken down affects the score. More granular statements provide a more detailed evaluation.
- **Context Relevance**: Faithfulness only measures if the answer is supported by the provided context, not whether the context is relevant to the question.

## Best Practices

- Use faithfulness alongside other metrics like context precision for a complete evaluation
- Ensure your LLM for evaluation is capable of detailed statement-by-statement analysis
- Review statement breakdowns to understand where hallucinations occur
- Consider the confidence level of each statement's support

For implementation examples, see the [usage examples](../usage/examples.md).