#!/usr/bin/env python3
"""
Example script to evaluate faithfulness in RAG systems.

Faithfulness measures whether the generated answer contains only factual claims 
that are supported by the retrieved context.

This script includes examples of both high-scoring and low-scoring cases.
"""
import instructor
from rag_evals import Faithfulness, FaithfulnessResult

# Initialize the LLM client
# For testing purposes we use the mock client, but in production you would use:
# client = instructor.from_provider("openai/gpt-4o-mini")
# Instead, we use a mock client that returns pre-defined results for testing
client = instructor.from_provider("openai/gpt-4o-mini")

# Example 1: High Faithfulness Score
# All statements in the answer are directly supported by the context
def test_high_faithfulness():
    question = "What are the benefits of exercise?"
    answer = "Regular exercise improves cardiovascular health and increases strength."
    context = [
        "Regular physical activity improves heart health and circulation.",
        "Weight training builds muscle strength and increases bone density.",
        "The earliest Olympic games were held in Ancient Greece."
    ]
    
    faithfulness_result = Faithfulness.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    print("HIGH FAITHFULNESS:")
    for stmt in faithfulness_result.statements:
        print(f"Supported: {stmt.is_supported}, Chunk IDs: {stmt.supporting_chunk_ids}")
    # Supported: True, Chunk IDs: [0]
    # Supported: True, Chunk IDs: [1]
    print(f"Faithfulness Score: {faithfulness_result.score}")
    # Faithfulness Score: 1.0
    
# Example 2: Mixed Faithfulness Score
# Some statements are supported, others are not
def test_mixed_faithfulness():
    question = "What are the benefits of meditation?"
    answer = "Meditation reduces stress, improves mental clarity, and cures all diseases."
    context = [
        "Regular meditation practice has been shown to reduce stress levels.",
        "Studies indicate meditation can improve mental clarity and focus.",
        "The history of meditation dates back thousands of years."
    ]
    faithfulness_result = Faithfulness.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    print("MIXED FAITHFULNESS:")
    for stmt in faithfulness_result.statements:
        print(f"Supported: {stmt.is_supported}, Chunk IDs: {stmt.supporting_chunk_ids}")
    # Supported: True, Chunk IDs: [0]
    # Supported: True, Chunk IDs: [1]
    # Supported: False, Chunk IDs: []
    print(f"Faithfulness Score: {faithfulness_result.score}")
    # Faithfulness Score: 0.6666666666666666
    
# Example 3: Low Faithfulness Score
# None of the statements are supported by the context
def test_low_faithfulness():
    question = "What is quantum computing?"
    answer = "Quantum computing uses subatomic particles to process information at incredible speeds, making it the fastest form of computing available today."
    context = [
        "Classical computing uses bits as the smallest unit of data.",
        "The history of modern computing begins with Charles Babbage and Ada Lovelace.",
        "Cloud computing allows users to access computing resources over the internet."
    ]
    
    # Expected result
    faithfulness_result = FaithfulnessResult(
        statements=[
            {
                "statement": "Quantum computing uses subatomic particles to process information",
                "is_supported": False,
                "supporting_chunk_ids": None
            },
            {
                "statement": "Quantum computing processes information at incredible speeds",
                "is_supported": False,
                "supporting_chunk_ids": None
            },
            {
                "statement": "Quantum computing is the fastest form of computing available today",
                "is_supported": False,
                "supporting_chunk_ids": None
            }
        ]
    )

    print("LOW FAITHFULNESS:")
    for stmt in faithfulness_result.statements:
        print(f"Supported: {stmt.is_supported}, Chunk IDs: {stmt.supporting_chunk_ids}")
    # Supported: False, Chunk IDs: None
    # Supported: False, Chunk IDs: None
    # Supported: False, Chunk IDs: None
    print(f"Faithfulness Score: {faithfulness_result.score}")
    # Faithfulness Score: 0.0

if __name__ == "__main__":
    test_high_faithfulness()
    test_mixed_faithfulness()
    test_low_faithfulness()