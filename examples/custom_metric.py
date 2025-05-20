#!/usr/bin/env python3
"""
Example script demonstrating how to create custom evaluation metrics in RAG Evals.

This script shows how to:
1. Define custom response models
2. Create custom evaluation metrics
3. Apply custom metrics to evaluate RAG system outputs
"""
import asyncio
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional
from rag_evals import base

# Initialize the LLM client
client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

# Define custom response models
class ReferenceAccuracyItem(BaseModel):
    """Evaluates a statement's accuracy with respect to citation/reference quality."""
    statement: str = Field(description="A factual claim from the answer")
    has_reference: bool = Field(description="Whether the statement includes a citation or reference")
    reference_quality: Optional[float] = Field(
        None,
        ge=0.0, 
        le=1.0,
        description="Quality score for the reference (0.0-1.0), if present"
    )
    explanation: str = Field(description="Explanation of the reference quality assessment")

class ReferenceAccuracyResult(BaseModel):
    """Overall result of reference accuracy evaluation."""
    statements: List[ReferenceAccuracyItem] = Field(description="List of statements with reference evaluations")
    
    @property
    def score(self) -> float:
        """Overall reference accuracy score based on statements with references."""
        statements_with_refs = [s for s in self.statements if s.has_reference]
        if not statements_with_refs:
            return 0.0
        return sum(s.reference_quality or 0.0 for s in statements_with_refs) / len(statements_with_refs)
    
    @property
    def reference_coverage(self) -> float:
        """Percentage of statements that have references."""
        if not self.statements:
            return 0.0
        return sum(1 for s in self.statements if s.has_reference) / len(self.statements)

# Create a custom evaluation metric
ReferenceAccuracy = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator assessing the quality of citations and references in a RAG system answer.
    
    ## Your Task:
    
    Given a user's question, a generated answer, and retrieved context chunks, evaluate:
    1. Which statements in the answer include citations or references
    2. How accurately those references point to information in the context
    3. The overall reference quality of the answer
    
    ## Evaluation Process:
    
    1. Identify all factual claims or statements in the answer
    2. For each statement, determine if it includes a citation or reference
    3. If a reference is present, assess its quality on a scale of 0.0-1.0:
       - 1.0: Perfect reference - clearly points to specific context that directly supports the statement
       - 0.8: Good reference - points to relevant context with minor issues
       - 0.5: Mediocre reference - points to somewhat relevant context but is vague or imprecise
       - 0.2: Poor reference - points to context that barely relates to the statement
       - 0.0: Incorrect reference - points to context that doesn't support the statement at all
    4. Provide an explanation for each reference quality assessment
    
    ## Output Format:
    
    Your output should be a structured JSON object following this format:
    {
        "statements": [
            {
                "statement": "The factual claim from the answer",
                "has_reference": true/false,
                "reference_quality": 0.0-1.0 (null if has_reference is false),
                "explanation": "Your reasoning for the reference quality assessment"
            },
            ...
        ]
    }
    
    ## Important Considerations:
    
    - A reference can be explicit (e.g., [1], [source]) or implicit (e.g., "according to the context")
    - Focus only on how well the references point to supporting evidence, not on factual accuracy
    - If a statement has no reference, set has_reference to false and reference_quality to null
    - Be specific in your explanations about why each reference deserves its quality score
    """,
    response_model=ReferenceAccuracyResult
)

# Sample evaluation data
question = "What are the health benefits of drinking green tea?"
answer = """
Green tea offers numerous health benefits. According to source [1], it contains antioxidants called catechins that may help prevent cell damage. Research [2] has shown that green tea can boost metabolism and aid in weight loss. Some studies indicate it may help reduce the risk of heart disease and certain cancers [3], though more research is needed. Green tea has been used in traditional medicine for centuries.
"""

context = [
    "Green tea contains polyphenols like epigallocatechin gallate (EGCG) and other catechins that act as powerful antioxidants, which may protect cells from damage.",
    "Some research suggests that the catechins in green tea can boost metabolic rate and increase fat oxidation, potentially aiding in weight management.",
    "Observational studies have found associations between green tea consumption and reduced risk of cardiovascular disease in some populations.",
    "Green tea originated in China and has been used in traditional medicine for thousands of years to treat various ailments."
]

async def evaluate_reference_accuracy():
    """Evaluate reference accuracy in a RAG system output."""
    print("Evaluating reference accuracy...")
    
    # Run the evaluation
    result = await ReferenceAccuracy.agrade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    # Print the results
    print("\n=== REFERENCE ACCURACY EVALUATION ===")
    print(f"Overall Reference Accuracy Score: {result.score:.2f}")
    print(f"Reference Coverage: {result.reference_coverage:.2f}")
    
    print("\nStatement Evaluations:")
    for i, stmt in enumerate(result.statements):
        print(f"\n[Statement {i+1}] {stmt.statement}")
        print(f"Has Reference: {stmt.has_reference}")
        if stmt.has_reference:
            print(f"Reference Quality: {stmt.reference_quality:.2f}")
        print(f"Explanation: {stmt.explanation}")
    
    return result

# Create a custom combined metrics evaluator
class CombinedMetrics(BaseModel):
    """Combined metrics evaluation result."""
    faithfulness: float = Field(description="Factual accuracy score (0.0-1.0)")
    reference_accuracy: float = Field(description="Reference quality score (0.0-1.0)")
    reference_coverage: float = Field(description="Proportion of statements with references (0.0-1.0)")
    
    @property
    def combined_score(self) -> float:
        """Weighted combined score of all metrics."""
        weights = {
            "faithfulness": 0.6,
            "reference_accuracy": 0.3,
            "reference_coverage": 0.1
        }
        return (
            weights["faithfulness"] * self.faithfulness +
            weights["reference_accuracy"] * self.reference_accuracy +
            weights["reference_coverage"] * self.reference_coverage
        )

async def main():
    # Evaluate reference accuracy
    ref_accuracy_result = await evaluate_reference_accuracy()
    
    # Normally, you would run other metrics here
    # For demonstration, we'll use a placeholder value for faithfulness
    faithfulness_score = 0.85  # This would come from a real evaluation
    
    # Create combined metrics
    combined = CombinedMetrics(
        faithfulness=faithfulness_score,
        reference_accuracy=ref_accuracy_result.score,
        reference_coverage=ref_accuracy_result.reference_coverage
    )
    
    # Print combined metrics
    print("\n=== COMBINED METRICS ===")
    print(f"Faithfulness: {combined.faithfulness:.2f}")
    print(f"Reference Accuracy: {combined.reference_accuracy:.2f}")
    print(f"Reference Coverage: {combined.reference_coverage:.2f}")
    print(f"Combined Score: {combined.combined_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())