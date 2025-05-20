#!/usr/bin/env python3
"""
Example script to evaluate answer relevance in RAG systems.

Answer Relevance evaluates how well the generated answer addresses the original question,
regardless of factual accuracy or context support.

This script includes examples of high, mixed, and low relevance scores.
"""
import instructor
from rag_evals.metrics.relevance import AnswerRelevance, RelevanceScore

# Initialize the LLM client
# For testing purposes we use the mock client, but in production you would use:
# client = instructor.from_provider("openai/gpt-4o-mini")
client = instructor.from_provider("openai/gpt-4o-mini")

# Example 1: High Relevance Score
# The answer directly, completely, and concisely addresses the question.
def test_high_relevance():
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    # Context is not used by AnswerRelevance but included for API consistency if needed elsewhere
    context = ["France is a country in Europe.", "Paris is known for the Eiffel Tower."]
    
    relevance_result = AnswerRelevance.grade(
        question=question,
        answer=answer,
        context=context, # Not used by this specific metric
        client=client
    )
    
    print("HIGH RELEVANCE:")
    print(f"Overall Score: {relevance_result.overall_score}")
    print(f"  Topical Match: {relevance_result.topical_match}")
    print(f"  Completeness: {relevance_result.completeness}")
    print(f"  Conciseness: {relevance_result.conciseness}")
    print(f"  Reasoning: {relevance_result.reasoning}")
    # Expected (ideal LLM):
    # Overall Score: 1.0
    #   Topical Match: 1.0
    #   Completeness: 1.0
    #   Conciseness: 1.0
    #   Reasoning: The answer directly states that the capital of France is Paris, which perfectly aligns with the question asked. It fully addresses the question's main aspect without any digressions or irrelevant information.

# Example 2: Mixed Relevance Score
# The answer is on topic but might be incomplete or not fully concise.
def test_mixed_relevance():
    question = "What are the benefits of regular exercise and a balanced diet?"
    answer = "Regular exercise improves cardiovascular health. A balanced diet is also good."
    context = ["Exercise helps the heart.", "Healthy eating provides nutrients."]

    relevance_result = AnswerRelevance.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )

    print("\nMIXED RELEVANCE:")
    print(f"Overall Score: {relevance_result.overall_score}")
    print(f"  Topical Match: {relevance_result.topical_match}")
    print(f"  Completeness: {relevance_result.completeness}")
    print(f"  Conciseness: {relevance_result.conciseness}")
    print(f"  Reasoning: {relevance_result.reasoning}")
    # Expected (ideal LLM might give something like this):
    # Overall Score: 0.5
    #   Topical Match: 0.7
    #   Completeness: 0.4
    #   Conciseness: 0.7
    #   Reasoning: The answer addresses the topic of benefits from regular exercise and a balanced diet, hence it has a good topical match (0.7). However, it only touches on a couple of benefits rather than providing a comprehensive list, leading to a completeness score of 0.4. The answer is fairly concise, focusing on the main points with minimal irrelevant information, leading to a score of 0.7 for conciseness. Overall, due to these scores, the average relevance score is 0.5.

# Example 3: Low Relevance Score
# The answer is off-topic or fails to address the question.
def test_low_relevance():
    question = "What is the process of photosynthesis?"
    answer = "The mitochondria is the powerhouse of the cell."
    context = ["Photosynthesis occurs in chloroplasts.", "Cellular respiration involves mitochondria."]
    
    relevance_result = AnswerRelevance.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )

    print("\nLOW RELEVANCE:")
    print(f"Overall Score: {relevance_result.overall_score}")
    print(f"  Topical Match: {relevance_result.topical_match}")
    print(f"  Completeness: {relevance_result.completeness}")
    print(f"  Conciseness: {relevance_result.conciseness}")
    print(f"  Reasoning: {relevance_result.reasoning}")
    # Expected (ideal LLM):
    # Overall Score: 0.0
    #   Topical Match: 0.0
    #   Completeness: 0.0
    #   Conciseness: 1.0
    #   Reasoning: The answer does not address the question about photosynthesis at all. Instead, it provides information about mitochondria, which is related to cellular respiration. Thus, the topical match is 0.0, as it is completely off-topic. The completeness score is also 0.0 because the answer fails to address any part of the question. However, the answer is concise in its statement, hence the conciseness score is 1.0.

if __name__ == "__main__":
    print("Running Answer Relevance Examples:")
    print("NOTE: Actual scores depend on the LLM's judgment and may vary.")
    print("The 'client' used here is a real LLM, not a fixed mock for these examples.")
    print("Therefore, scores might not exactly match comments if the LLM has a different interpretation.")
    
    test_high_relevance()
    test_mixed_relevance()
    test_low_relevance() 