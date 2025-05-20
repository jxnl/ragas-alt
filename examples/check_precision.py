#!/usr/bin/env python3
"""
Example script to evaluate context precision in RAG systems.

Context Precision (also known as Chunk Relevancy) measures whether each 
retrieved context chunk is relevant to the original question.

This script includes examples of high, mixed, and low precision scores.
"""
import instructor
from rag_evals.metrics.precision import ChunkPrecision
from rag_evals.base import ChunkGradedBinary, ChunkBinaryScore # Added for explicit result construction
import asyncio

# Initialize the LLM client
# For testing purposes we use the mock client, but in production you would use:
# client = instructor.from_provider("openai/gpt-4o-mini")
# Instead, we use a mock client that returns pre-defined results for testing
client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

# Example 1: High Precision Score
# All retrieved context chunks are relevant to the question.
async def test_high_precision():
    question = "What are the primary colors?"
    # Answer is not strictly needed for ChunkPrecision but often available
    answer = "The primary colors are red, yellow, and blue." 
    context = [
        "Red is a primary color.",
        "Yellow is a primary color.",
        "Blue is a primary color.",
        "Green is a secondary color, made by mixing blue and yellow."
    ]
    
    precision_result = await ChunkPrecision.grade(
        question=question,
        answer=answer, # Though not used by the metric, it's part of the signature
        context=context,
        client=client
    )
    
    print("HIGH PRECISION:")
    for chunk_score in precision_result.graded_chunks:
        print(f"Chunk ID: {chunk_score.id_chunk}, Relevant: {chunk_score.score}")
    # Expected:
    # Chunk ID: 0, Relevant: True
    # Chunk ID: 1, Relevant: True
    # Chunk ID: 2, Relevant: True
    # Chunk ID: 3, Relevant: False (discusses a secondary color, less directly relevant to "primary colors")
    # Adjusting expectation based on typical LLM behavior for strict relevance to "primary colors"
    # For this mock, let's assume it correctly identifies the first three. The actual LLM might vary.
    
    # For deterministic testing with a mock, we'd ideally mock the LLM's response.
    # Here, we'll simulate an expected good result based on a capable LLM.
    # For the purpose of this example, we'll assume the LLM correctly identifies the first 3 as relevant.
    # The 4th chunk about green might be considered less relevant if strictly focusing on "primary colors".
    # Let's assume for this example, it's also considered relevant as it provides context around primary/secondary.
    # Actual LLM behavior will determine this.

    # To make this testable without a live LLM or complex mocking for this example,
    # let's assume a result where the first 3 are true and the last is false for demonstration.
    # If client were a mock, we'd pre-program its response.
    # Since it's not, we'll just print what an ideal LLM might return.
    # For this illustrative script, we'll show how to interpret the result.
    # A real test would compare against expected ChunkGradedBinary.

    print(f"Precision Score: {precision_result.score}")
    # Expected: (3 relevant / 4 total) = 0.75 if chunk 3 is False
    # Or 1.0 if all are considered relevant by the LLM. Let's proceed with the former.

# Example 2: Mixed Precision Score
# Some retrieved chunks are relevant, others are not.
async def test_mixed_precision():
    question = "Tell me about the Eiffel Tower."
    answer = "The Eiffel Tower is a famous landmark in Paris."
    context = [
        "The Eiffel Tower is located on the Champ de Mars in Paris, France.", # Relevant
        "It was designed and built by Gustave Eiffel's company.", # Relevant
        "The tower is 330 meters tall.", # Relevant
        "Paris is also known for the Louvre Museum, home to the Mona Lisa.", # Less relevant
        "The primary colors are red, yellow, and blue." # Irrelevant
    ]
    precision_result = await ChunkPrecision.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    print("\nMIXED PRECISION:")
    for chunk_score in precision_result.graded_chunks:
        print(f"Chunk ID: {chunk_score.id_chunk}, Relevant: {chunk_score.score}")
    # Expected:
    # Chunk ID: 0, Relevant: True
    # Chunk ID: 1, Relevant: True
    # Chunk ID: 2, Relevant: True
    # Chunk ID: 3, Relevant: False (or True, depending on LLM's interpretation of "about")
    # Chunk ID: 4, Relevant: False
    print(f"Precision Score: {precision_result.score}")
    # Expected: (3 relevant / 5 total) = 0.6 (if chunk 3 is False)
    # Or (4 relevant / 5 total) = 0.8 (if chunk 3 is True)

# Example 3: Low Precision Score
# Most or all retrieved chunks are not relevant to the question.
async def test_low_precision():
    question = "What is the capital of Japan?"
    answer = "The capital of Japan is Tokyo."
    context = [
        "Mount Fuji is the tallest mountain in Japan.", # Not directly capital
        "Sushi is a popular Japanese dish.", # Not capital
        "The Japanese currency is the Yen.", # Not capital
        "Samurai were the military nobility of pre-industrial Japan." # Not capital
    ]
    
    # For a fully deterministic example without relying on LLM call for grading:
    # We can construct the expected result directly if we were testing the logic AFTER grading
    # However, the .grade() call invokes the LLM.
    # For this example, we'll call .grade() and print the result.
    # In a real unit test, you would mock 'client.chat.completions.create'.
    
    precision_result = await ChunkPrecision.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )

    print("\nLOW PRECISION:")
    for chunk_score in precision_result.graded_chunks:
        print(f"Chunk ID: {chunk_score.id_chunk}, Relevant: {chunk_score.score}")
    # Expected (ideal LLM):
    # Chunk ID: 0, Relevant: False
    # Chunk ID: 1, Relevant: False
    # Chunk ID: 2, Relevant: False
    # Chunk ID: 3, Relevant: False
    print(f"Precision Score: {precision_result.score}")
    # Expected: 0.0

async def main():
    await test_high_precision()
    await test_mixed_precision()
    await test_low_precision() 

if __name__ == "__main__":
    asyncio.run(main()) 