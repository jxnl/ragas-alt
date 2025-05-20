# RAG Evals Examples

This page provides practical examples of using RAG Evals in different scenarios.

## Basic Usage Example

This example shows how to evaluate a simple RAG system using faithfulness, context precision, and answer relevance metrics:

```python
import instructor
from rag_evals import Faithfulness, ChunkPrecision, AnswerRelevance

# Initialize with LLM
client = instructor.from_provider("openai/gpt-4o-mini")

# Sample RAG output
question = "What are the benefits of exercise?"
answer = "Regular exercise improves cardiovascular health and increases strength."
context = [
    "Regular physical activity improves heart health and circulation.",
    "Weight training builds muscle strength and increases bone density.",
    "The earliest Olympic games were held in Ancient Greece."
]

# Evaluate faithfulness
faithfulness_result = Faithfulness.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)

# Evaluate context precision
precision_result = ChunkPrecision.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)

# Evaluate answer relevance
relevance_result = AnswerRelevance.grade(
    question=question,
    answer=answer,
    context=context,  # Context isn't used in relevance evaluation
    client=client
)

# Process and display results
print(f"Overall Faithfulness Score: {faithfulness_result.overall_faithfulness_score:.2f}")
print(f"Overall Context Precision: {precision_result.avg_score:.2f}")
print(f"Overall Answer Relevance: {relevance_result.overall_score:.2f}")

# Detailed analysis
print("\nFaithfulness Breakdown:")
for statement in faithfulness_result.statements:
    print(f"- {statement.statement}")
    print(f"  Supported: {statement.is_supported}")
    if statement.is_supported and statement.supporting_chunk_ids:
        print(f"  Supporting chunks: {statement.supporting_chunk_ids}")

print("\nContext Precision Breakdown:")
for chunk in precision_result.graded_chunks:
    print(f"- Chunk {chunk.id_chunk}: {'Relevant' if chunk.score else 'Not Relevant'}")

print("\nAnswer Relevance Breakdown:")
print(f"- Topical Match: {relevance_result.topical_match:.2f}")
print(f"- Completeness: {relevance_result.completeness:.2f}")
print(f"- Conciseness: {relevance_result.conciseness:.2f}")
print(f"- Reasoning: {relevance_result.reasoning}")
```

## Parallel Evaluation Example

This example demonstrates how to run multiple evaluations in parallel for better performance:

```python
import asyncio
import instructor
from rag_evals import Faithfulness, ChunkPrecision, AnswerRelevance

# Initialize with async client
async_client = instructor.AsyncInstructor(provider="openai/gpt-4o-mini")

# Sample RAG outputs to evaluate
examples = [
    {
        "question": "What are the benefits of exercise?",
        "answer": "Regular exercise improves cardiovascular health and increases strength.",
        "context": [
            "Regular physical activity improves heart health and circulation.",
            "Weight training builds muscle strength and increases bone density.",
            "The earliest Olympic games were held in Ancient Greece."
        ]
    },
    {
        "question": "How does photosynthesis work?",
        "answer": "Photosynthesis is the process by which plants convert sunlight into energy.",
        "context": [
            "Photosynthesis is the process by which plants use sunlight to synthesize foods from carbon dioxide and water.",
            "The process primarily happens in the plant's leaves through their chloroplasts.",
            "Solar panels are designed to convert sunlight into electricity."
        ]
    }
]

async def evaluate_example(example):
    """Evaluate a single example using multiple metrics in parallel"""
    faithfulness_task = Faithfulness.agrade(
        question=example["question"],
        answer=example["answer"],
        context=example["context"],
        client=async_client
    )
    
    precision_task = ChunkPrecision.agrade(
        question=example["question"],
        answer=example["answer"],
        context=example["context"],
        client=async_client
    )
    
    relevance_task = AnswerRelevance.agrade(
        question=example["question"],
        answer=example["answer"],
        context=example["context"],
        client=async_client
    )
    
    # Run all evaluations in parallel
    faithfulness_result, precision_result, relevance_result = await asyncio.gather(
        faithfulness_task, precision_task, relevance_task
    )
    
    return {
        "question": example["question"],
        "faithfulness_score": faithfulness_result.overall_faithfulness_score,
        "precision_score": precision_result.avg_score,
        "relevance_score": relevance_result.overall_score
    }

async def evaluate_all_examples():
    """Evaluate all examples in parallel"""
    tasks = [evaluate_example(example) for example in examples]
    results = await asyncio.gather(*tasks)
    return results

# Run the evaluations
results = asyncio.run(evaluate_all_examples())

# Display results
for result in results:
    print(f"\nQuestion: {result['question']}")
    print(f"Faithfulness Score: {result['faithfulness_score']:.2f}")
    print(f"Precision Score: {result['precision_score']:.2f}")
    print(f"Relevance Score: {result['relevance_score']:.2f}")
```

## Custom Prompt Example

This example shows how to customize the evaluation prompts:

```python
from rag_evals import base, FaithfulnessResult, RelevanceScore, ChunkGradedBinary

# Define a custom answer relevance evaluator with a modified prompt
CustomRelevance = base.ContextEvaluation(
    prompt="""
    You are evaluating how well an answer addresses the question.
    
    Evaluate the answer along these dimensions:
    1. Topical relevance (0-1): Does the answer address the topic of the question?
    2. Completeness (0-1): Does the answer cover all parts of the question?
    3. Conciseness (0-1): Is the answer appropriately detailed without unnecessary information?
    
    Give each dimension a score and provide your reasoning.
    """,
    response_model=RelevanceScore
)

# Define a custom context precision evaluator
CustomPrecision = base.ContextEvaluation(
    prompt="""
    You are evaluating the relevance of retrieved context chunks to the original question.
    
    For each retrieved chunk:
    1. Determine if it contains information that would help answer the question
    2. Consider both direct relevance (explicitly addresses the question) and indirect relevance (provides background or related information)
    3. Be moderately strict - the chunk should contain actual helpful information, not just be on a vaguely related topic
    
    Output a binary judgment (relevant/not relevant) for the chunk.
    """,
    response_model=base.ChunkGradedBinary
)

# Use the custom evaluators
relevance_result = CustomRelevance.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)

precision_result = CustomPrecision.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)
```

## Comprehensive Evaluation Example

This example demonstrates a complete evaluation framework that assesses all key relationships:

```python
import pandas as pd
import instructor
from rag_evals import Faithfulness, ChunkPrecision, AnswerRelevance

def evaluate_rag_system(question, context, answer, client):
    """Comprehensive RAG evaluation across all key dimensions"""
    
    # 1. Context → Question: Context Relevance
    precision_result = ChunkPrecision.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    # 2. Context → Answer: Faithfulness
    faithfulness_result = Faithfulness.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    # 3. Question → Answer: Answer Relevance
    relevance_result = AnswerRelevance.grade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    # Calculate aggregate scores
    retriever_score = precision_result.avg_score
    generator_score = faithfulness_result.overall_faithfulness_score
    end_to_end_score = relevance_result.overall_score
    
    # Overall system score (simple average)
    overall_score = (retriever_score + generator_score + end_to_end_score) / 3
    
    return {
        "overall_score": overall_score,
        "retriever_score": retriever_score,
        "generator_score": generator_score,
        "end_to_end_score": end_to_end_score,
        "precision_details": precision_result,
        "faithfulness_details": faithfulness_result,
        "relevance_details": relevance_result
    }

# Example usage
client = instructor.from_provider("openai/gpt-4o-mini")

question = "What are the health benefits of meditation?"
answer = "Meditation reduces stress and improves mental focus. It also helps with depression."
context = [
    "Regular meditation reduces stress and anxiety.",
    "Meditation can improve focus and attention span.",
    "The history of meditation dates back to ancient civilizations."
]

results = evaluate_rag_system(question, context, answer, client)

# Display results
print(f"Overall System Score: {results['overall_score']:.2f}")
print(f"Retriever Score (Context Relevance): {results['retriever_score']:.2f}")
print(f"Generator Score (Faithfulness): {results['generator_score']:.2f}")
print(f"End-to-End Score (Answer Relevance): {results['end_to_end_score']:.2f}")

# Detailed analysis (example)
print("\nUnfaithful statements:")
for stmt in results['faithfulness_details'].statements:
    if not stmt.is_supported:
        print(f"- {stmt.statement}")

print("\nIrrelevant context chunks:")
for chunk in results['precision_details'].graded_chunks:
    if not chunk.score:
        print(f"- Chunk {chunk.id_chunk}: {context[chunk.id_chunk][:50]}...")
```

These examples demonstrate the flexibility and power of RAG Evals for different evaluation scenarios. You can adapt these patterns to fit your specific evaluation needs.