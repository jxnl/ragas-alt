#!/usr/bin/env python3
"""
Example script demonstrating batch evaluation of multiple questions using RAG Evals.

This script shows how to:
1. Process multiple question-answer-context sets efficiently
2. Use asyncio to parallelize evaluations
3. Aggregate and analyze results across multiple evaluations
"""
import asyncio
import instructor
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from rag_evals import Faithfulness, ChunkPrecision, AnswerRelevance

# Initialize the async LLM client
# For production, use an actual OpenAI client:
client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

# Sample evaluation data
evaluation_data = [
    {
        "id": "q1",
        "question": "What are the benefits of regular exercise?",
        "answer": "Regular exercise improves cardiovascular health, increases strength, and can help with weight management.",
        "context": [
            "Regular physical activity improves heart health and circulation.",
            "Weight training builds muscle strength and increases bone density.",
            "Exercise helps control weight by burning calories and increasing metabolism."
        ]
    },
    {
        "id": "q2",
        "question": "How does photosynthesis work?",
        "answer": "Photosynthesis is the process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
        "context": [
            "Photosynthesis occurs in plant chloroplasts, primarily in leaf cells.",
            "The process converts light energy into chemical energy in the form of glucose.",
            "Carbon dioxide and water are the raw materials, while oxygen is released as a byproduct."
        ]
    },
    {
        "id": "q3",
        "question": "Who was Marie Curie?",
        "answer": "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity, discovered polonium and radium, and was the first woman to win a Nobel Prize.",
        "context": [
            "Marie Curie (1867-1934) was a Polish-born physicist and chemist.",
            "She conducted groundbreaking research on radioactivity, a term she coined.",
            "Curie discovered the elements polonium and radium with her husband Pierre.",
            "She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in multiple scientific fields (Physics and Chemistry)."
        ]
    }
]

async def evaluate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single question-answer-context item using multiple metrics."""
    question = item["question"]
    answer = item["answer"]
    context = item["context"]
    
    # Run multiple evaluations in parallel
    faithfulness_task = Faithfulness.agrade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    precision_task = ChunkPrecision.agrade(
        question=question,
        answer=None,  # Precision only needs question and context
        context=context,
        client=client
    )
    
    relevance_task = AnswerRelevance.agrade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    # Await all tasks
    faithfulness_result, precision_result, relevance_result = await asyncio.gather(
        faithfulness_task, precision_task, relevance_task
    )
    
    # Compile results
    return {
        "id": item["id"],
        "question": question,
        "faithfulness_score": faithfulness_result.score,
        "precision_score": precision_result.score,
        "relevance_score": relevance_result.overall_score,
        "statement_count": len(faithfulness_result.statements),
        "context_chunks": len(context),
        # Store the detailed results for further analysis if needed
        "detailed_results": {
            "faithfulness": faithfulness_result,
            "precision": precision_result,
            "relevance": relevance_result
        }
    }

async def batch_evaluate(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run batch evaluation on multiple items in parallel."""
    tasks = [evaluate_item(item) for item in data]
    return await asyncio.gather(*tasks)

def analyze_results(results: List[Dict[str, Any]]) -> None:
    """Analyze and present evaluation results."""
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            "id": r["id"],
            "question": r["question"],
            "faithfulness": r["faithfulness_score"],
            "precision": r["precision_score"],
            "relevance": r["relevance_score"],
            "statement_count": r["statement_count"],
            "context_chunks": r["context_chunks"]
        } 
        for r in results
    ])
    
    # Print summary statistics
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total items evaluated: {len(df)}")
    print("\nAverage scores:")
    print(f"  Faithfulness: {df['faithfulness'].mean():.2f}")
    print(f"  Precision: {df['precision'].mean():.2f}")
    print(f"  Relevance: {df['relevance'].mean():.2f}")
    print("\nScore ranges:")
    print(f"  Faithfulness: {df['faithfulness'].min():.2f} - {df['faithfulness'].max():.2f}")
    print(f"  Precision: {df['precision'].min():.2f} - {df['precision'].max():.2f}")
    print(f"  Relevance: {df['relevance'].min():.2f} - {df['relevance'].max():.2f}")
    
    # Print detailed results
    print("\n=== DETAILED RESULTS ===")
    print(df.to_string())
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"batch_evaluation_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")

async def main():
    print("Starting batch evaluation of RAG performance metrics...")
    start_time = datetime.now()
    
    results = await batch_evaluate(evaluation_data)
    analyze_results(results)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nEvaluation completed in {duration:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())