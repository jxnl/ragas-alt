# Customizing RAG Evals

RAG Evals is designed to be flexible and customizable. This guide explains how to tailor the evaluation process to your specific needs.

## Customizing Prompts

The most straightforward way to customize RAG Evals is by modifying the evaluation prompts:

```python
from rag_evals import base
from rag_evals.score_faithfulness import FaithfulnessResult

# Access the original prompt
from rag_evals.score_faithfulness import Faithfulness
original_prompt = Faithfulness.prompt

# Create a customized evaluator with your own prompt
CustomFaithfulness = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator assessing the factual accuracy of answers.
    
    For each statement in the answer:
    1. Determine if it is fully supported by the context
    2. Be extremely strict - only mark as supported if there is explicit evidence
    3. Identify which context chunks provide the supporting evidence
    
    Your evaluation should be thorough and rigorous.
    """,
    response_model=FaithfulnessResult
)
```

### Prompt Design Tips

When customizing prompts, consider these best practices:

1. **Be Explicit**: Clearly define what constitutes a supported vs. unsupported statement
2. **Set the Evaluation Tone**: Indicate how strict the evaluation should be
3. **Specify Output Format**: Ensure the prompt produces output compatible with the response model
4. **Provide Examples**: For complex evaluations, include examples in the prompt

## Customizing Response Models

You can also create custom response models for more specialized evaluations:

```python
from pydantic import BaseModel, Field
from rag_evals import base

# Define a custom response model
class EnhancedStatementEvaluation(BaseModel):
    statement: str = Field(description="An individual claim from the answer")
    is_supported: bool = Field(description="Whether the statement is supported")
    supporting_chunk_ids: list[int] = Field(default_factory=list)
    confidence_level: str = Field(
        description="Confidence level of the support assessment",
        default="high"
    )

class EnhancedFaithfulnessResult(BaseModel):
    statements: list[EnhancedStatementEvaluation]
    
    @property
    def overall_faithfulness_score(self) -> float:
        if not self.statements:
            return 0.0
        supported_statements = sum(s.is_supported for s in self.statements)
        return supported_statements / len(self.statements)
    
    @property
    def high_confidence_score(self) -> float:
        """Score based only on high-confidence assessments"""
        high_confidence = [s for s in self.statements if s.confidence_level == "high"]
        if not high_confidence:
            return 0.0
        supported = sum(s.is_supported for s in high_confidence)
        return supported / len(high_confidence)

# Create an evaluator that uses this model
EnhancedFaithfulness = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator assessing the factual accuracy of answers.
    
    For each statement in the answer:
    1. Determine if it is supported by the context
    2. Assign a confidence level (high, medium, low) to your assessment
    3. Identify which context chunks provide the supporting evidence
    
    Output a structured evaluation including all these elements.
    """,
    response_model=EnhancedFaithfulnessResult
)
```

## Creating New Metrics

You can create entirely new evaluation metrics by defining new evaluation classes:

```python
from pydantic import BaseModel, Field
from rag_evals import base

# Define a response model for answer relevancy
class RelevancyResult(BaseModel):
    topical_match: float = Field(ge=0.0, le=1.0, description="Score for how well the answer topic matches the question")
    completeness: float = Field(ge=0.0, le=1.0, description="Score for how completely the answer addresses all aspects of the question")
    specificity: float = Field(ge=0.0, le=1.0, description="Score for how specific and detailed the answer is")
    
    @property
    def overall_relevancy_score(self) -> float:
        # Weighted average of the component scores
        return (0.4 * self.topical_match + 
                0.4 * self.completeness + 
                0.2 * self.specificity)

# Create the relevancy evaluator
AnswerRelevancy = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator assessing how well an answer addresses the original question.
    
    Evaluate the answer along three dimensions:
    
    1. Topical Match (0.0-1.0): How well does the answer's subject matter align with what was asked?
    2. Completeness (0.0-1.0): How thoroughly does the answer address all aspects of the question?
    3. Specificity (0.0-1.0): How specific and detailed is the answer relative to what was asked?
    
    Provide scores for each dimension and explain your reasoning.
    """,
    response_model=RelevancyResult
)
```

## Customizing the Evaluation Process

The `ContextEvaluation` class provides a flexible foundation for custom evaluation processes:

### Custom Templates

You can customize the chunk template used for evaluation:

```python
from rag_evals import base
from textwrap import dedent

# Create an evaluator with a custom template
CustomEvaluator = base.ContextEvaluation(
    prompt="Your evaluation prompt here...",
    response_model=YourResponseModel,
    chunk_template=dedent("""
        <custom_evaluation>
            <query>{{ question }}</query>
            <response>{{ answer }}</response>
            <retrieved_documents>
                {% for chunk in context %}
                <document id="{{ chunk.id }}">
                    {{ chunk.chunk }}
                </document>
                {% endfor %}
            </retrieved_documents>
        </custom_evaluation>
    """)
)
```

### Adding Examples

You can provide examples to guide the evaluation:

```python
# Define examples
examples = [
    {
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis is the process by which plants convert sunlight into energy.",
        "context": ["Photosynthesis is the process by which plants use sunlight to create energy."],
        "expected_result": {"faithfulness_score": 1.0}
    },
    {
        "question": "Who was Albert Einstein?",
        "answer": "Einstein was a physicist who developed the theory of relativity and won a Nobel Prize.",
        "context": ["Albert Einstein was a theoretical physicist known for developing the theory of relativity."],
        "expected_result": {"faithfulness_score": 0.5}
    }
]

# Create an evaluator with examples
ExampleGuidedEvaluator = base.ContextEvaluation(
    prompt="Your evaluation prompt here...",
    response_model=YourResponseModel,
    examples=examples
)
```

## Integration with Custom LLMs

You can use any Instructor-compatible LLM provider for evaluations:

```python
import instructor
from rag_evals.score_faithfulness import Faithfulness

# Using OpenAI
openai_client = instructor.from_provider("openai/gpt-4-turbo")

# Using Anthropic
anthropic_client = instructor.from_provider("anthropic/claude-3-opus")

# Using Cohere
cohere_client = instructor.from_provider("cohere/command-r")

# Using a local model
local_client = instructor.from_provider("local/llama-3-70b")

# Run evaluation with your preferred client
result = Faithfulness.grade(
    question=question,
    answer=answer,
    context=context,
    client=your_client_of_choice
)
```

By leveraging these customization options, you can adapt RAG Evals to a wide range of evaluation needs and scenarios.