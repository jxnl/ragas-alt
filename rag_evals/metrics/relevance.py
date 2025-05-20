from __future__ import annotations
import logging
from pydantic import BaseModel, Field
from .. import base

logger = logging.getLogger(__name__)

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

AnswerRelevance = base.ContextEvaluation(
    prompt = """
    You are an expert evaluator assessing how well an answer addresses the original question.

    ## Your Task:
    Given a user's **question** and a generated **answer**, evaluate how relevant the answer is to the question, regardless of factual accuracy. Focus solely on whether the answer addresses what was asked.

    ## Evaluation Dimensions:
    Evaluate the answer along three dimensions:

    1. **Topical Match** (0.0-1.0): How well does the answer's subject matter align with what was asked?
       - 1.0: Perfect match, directly addresses the question topic
       - 0.7: Good match, mostly on topic but with minor digressions
       - 0.4: Partial match, partially addresses the question topic
       - 0.0: No match, completely off-topic

    2. **Completeness** (0.0-1.0): How thoroughly does the answer address all aspects of the question?
       - 1.0: Comprehensive, addresses all parts of the question fully
       - 0.7: Mostly complete, addresses main aspects but misses minor points
       - 0.4: Partially complete, addresses some aspects but misses important elements
       - 0.0: Incomplete, fails to address most aspects of the question

    3. **Conciseness** (0.0-1.0): How focused is the answer without irrelevant information?
       - 1.0: Perfectly concise, focused entirely on answering the question
       - 0.7: Mostly concise, with minimal irrelevant information
       - 0.4: Somewhat verbose, contains notable irrelevant information
       - 0.0: Extremely verbose, mostly irrelevant to the question

    ## Output Format:
    Provide an overall relevance score (average of the three dimensions), along with scores for each dimension and your reasoning. The output should follow this structure:

    {
      "overall_score": float,
      "topical_match": float,
      "completeness": float, 
      "conciseness": float,
      "reasoning": "Your explanation of the scores"
    }

    ## Important Considerations:
    - Focus solely on relevance to the question, not factual accuracy or correctness
    - Consider all aspects of multi-part questions
    - Be objective and consistent in your scoring
    - Provide clear reasoning for your scores
    """,
    response_model = RelevanceScore
)