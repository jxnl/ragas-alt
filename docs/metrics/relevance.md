# Answer Relevance Evaluation

Answer Relevance evaluates how well the generated answer addresses the original question. This metric helps assess the quality of the answer from the user's perspective, focusing on how useful and on-topic the response is.

## What Answer Relevance Measures

- **Definition**: Answer Relevance assesses how well the generated answer addresses the original question, regardless of whether the information in the answer is factually accurate or supported by the context.
- **Focus**: The relationship is (Answer → Question).
- **Purpose**: To determine if the answer is responsive to the user's information need, even if the retrieval or generation components have issues.

## How It Works

The Answer Relevance evaluator:

1. Examines the question and answer independently of the context
2. Evaluates how well the answer addresses the question along multiple dimensions
3. Provides a holistic score indicating the relevance of the answer to the question

## Implementation Details

The answer relevance implementation evaluates three key dimensions:

```python
from rag_evals.score_relevance import AnswerRelevance

relevance_result = AnswerRelevance.grade(
    question=question,
    answer=answer,
    context=context,  # Note: context is not used in this evaluation
    client=client
)
```

### Response Model

The `RelevanceScore` class contains:

```python
class RelevanceScore(BaseModel):
    """Represents the evaluation of answer relevance to the question."""
    overall_score: float  # Overall relevance score from 0-1
    topical_match: float  # How well the answer's topic matches the question
    completeness: float  # How completely the answer addresses all aspects
    conciseness: float  # How concise and focused the answer is
    reasoning: str  # Explanation of the reasoning behind the scores
```

### Evaluation Dimensions

1. **Topical Match**: How well does the answer's subject matter align with what was asked?
2. **Completeness**: How thoroughly does the answer address all aspects of the question?
3. **Conciseness**: How focused is the answer without irrelevant information?

The overall score is calculated as the average of these three dimensions.

### Example Output

```python
# Result example
relevance_result = RelevanceScore(
    overall_score=0.8,
    topical_match=0.9,
    completeness=0.7,
    conciseness=0.8,
    reasoning="The answer directly addresses the question about meditation benefits, covering most key aspects like stress reduction and improved focus. It's concise and stays on topic with minimal tangents."
)
```

## Customizing the Evaluation

You can customize the answer relevance prompt to adjust the evaluation criteria:

```python
from rag_evals import base
from rag_evals.score_relevance import RelevanceScore

# Create a customized evaluator with a modified prompt
CustomRelevance = base.ContextEvaluation(
    prompt="Your custom prompt here with specific evaluation criteria...",
    response_model=RelevanceScore
)
```

## Answer Relevance vs. Faithfulness

It's important to understand the difference between:

- **Answer Relevance**: Measures how well the answer addresses the question, regardless of factual accuracy
- **Faithfulness**: Measures how well the answer is supported by the context

For example, an answer can be highly relevant to the question but unfaithful to the context, or vice versa.

## Considerations When Using Answer Relevance

- **Multi-part Questions**: Consider how well the answer addresses all aspects of complex questions
- **Implicit Questions**: Evaluate how well the answer addresses the intent behind the question
- **Overly Verbose Answers**: Consider penalizing answers that include significant irrelevant information
- **Different Answer Styles**: Some questions may be appropriately answered with different levels of detail

## Best Practices

- Use answer relevance alongside other metrics like faithfulness and context precision
- Analyze low-scoring answers to understand where the response generation needs improvement
- Consider different weightings of topical match, completeness, and conciseness based on your use case
- Keep in mind that context-less answer relevance doesn't capture factual accuracy

For implementation examples, see the [usage examples](../usage/examples.md).