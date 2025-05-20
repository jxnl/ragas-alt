#!/usr/bin/env python3
"""
Example script demonstrating the use of utility functions for processing evaluation results.

This script shows how to:
1. Use the EvaluationResult and BatchEvaluationResults classes
2. Process evaluation results with utility functions
3. Generate statistics and visualizations from results
"""
import asyncio
import instructor
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

from rag_evals import Faithfulness, ChunkPrecision, AnswerRelevance
from rag_evals.utils import (
    EvaluationResult,
    BatchEvaluationResults,
    batch_evaluate,
    combine_metrics,
    evaluate_summary_statistics,
    filter_results
)

# Sample data
evaluation_data = [
    {
        "id": "q1",
        "question": "What are the benefits of regular exercise?",
        "answer": "Regular exercise improves cardiovascular health, increases strength, and can help with weight management.",
        "context": [
            "Regular physical activity improves heart health and circulation.",
            "Weight training builds muscle strength and increases bone density.",
            "Exercise helps control weight by burning calories and increasing metabolism."
        ],
        "metadata": {
            "category": "health",
            "difficulty": "easy"
        }
    },
    {
        "id": "q2",
        "question": "How does photosynthesis work?",
        "answer": "Photosynthesis is the process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
        "context": [
            "Photosynthesis occurs in plant chloroplasts, primarily in leaf cells.",
            "The process converts light energy into chemical energy in the form of glucose.",
            "Carbon dioxide and water are the raw materials, while oxygen is released as a byproduct."
        ],
        "metadata": {
            "category": "science",
            "difficulty": "medium"
        }
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
        ],
        "metadata": {
            "category": "history",
            "difficulty": "medium"
        }
    }
]

# Initialize the LLM client
client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

async def evaluate_faithfulness(item: Dict[str, Any]) -> EvaluationResult:
    """Evaluate faithfulness for a single item."""
    question = item["question"]
    answer = item["answer"]
    context = item["context"]
    metadata = item.get("metadata", {})
    
    # Run the evaluation
    faithfulness_result = await Faithfulness.agrade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    # Create and return an EvaluationResult
    return EvaluationResult(
        question=question,
        answer=answer,
        context=context,
        result=faithfulness_result,
        metadata=metadata
    )

async def evaluate_precision(item: Dict[str, Any]) -> EvaluationResult:
    """Evaluate precision for a single item."""
    question = item["question"]
    context = item["context"]
    metadata = item.get("metadata", {})
    
    # Run the evaluation
    precision_result = await ChunkPrecision.agrade(
        question=question,
        answer=None,  # Precision only needs question and context
        context=context,
        client=client
    )
    
    # Create and return an EvaluationResult
    return EvaluationResult(
        question=question,
        answer=item.get("answer"),
        context=context,
        result=precision_result,
        metadata=metadata
    )

async def evaluate_relevance(item: Dict[str, Any]) -> EvaluationResult:
    """Evaluate relevance for a single item."""
    question = item["question"]
    answer = item["answer"]
    context = item["context"]
    metadata = item.get("metadata", {})
    
    # Run the evaluation
    relevance_result = await AnswerRelevance.agrade(
        question=question,
        answer=answer,
        context=context,
        client=client
    )
    
    # Create and return an EvaluationResult
    return EvaluationResult(
        question=question,
        answer=answer,
        context=context,
        result=relevance_result,
        metadata=metadata
    )

def plot_metric_comparison(metrics_results: Dict[str, BatchEvaluationResults]):
    """Plot comparison of multiple metrics."""
    metric_names = list(metrics_results.keys())
    
    # Prepare data for plotting
    data = {}
    for metric_name, batch_results in metrics_results.items():
        data[metric_name] = batch_results.scores
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    box = ax.boxplot(
        [data[m] for m in metric_names],
        patch_artist=True,
        labels=metric_names
    )
    
    # Color boxplots
    colors = ['lightblue', 'lightgreen', 'lightpink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add individual data points
    for i, metric in enumerate(metric_names):
        # Add jitter to x-position
        x = np.random.normal(i+1, 0.05, len(data[metric]))
        ax.scatter(x, data[metric], alpha=0.6)
    
    # Set plot labels and title
    ax.set_title('Comparison of Evaluation Metrics')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f"metric_comparison_{timestamp}.png")
    plt.close()
    
    return f"metric_comparison_{timestamp}.png"

async def main():
    print("Starting evaluation with utility functions...")
    
    # Process evaluations in batches
    print("Evaluating faithfulness...")
    faithfulness_results = await batch_evaluate(
        evaluation_func=evaluate_faithfulness,
        data=evaluation_data,
        batch_size=2,
        delay_seconds=0.5
    )
    
    print("Evaluating precision...")
    precision_results = await batch_evaluate(
        evaluation_func=evaluate_precision,
        data=evaluation_data,
        batch_size=2,
        delay_seconds=0.5
    )
    
    print("Evaluating relevance...")
    relevance_results = await batch_evaluate(
        evaluation_func=evaluate_relevance,
        data=evaluation_data,
        batch_size=2,
        delay_seconds=0.5
    )
    
    # Create BatchEvaluationResults containers
    faithfulness_batch = BatchEvaluationResults(
        results=faithfulness_results,
        evaluation_type="faithfulness"
    )
    
    precision_batch = BatchEvaluationResults(
        results=precision_results,
        evaluation_type="precision"
    )
    
    relevance_batch = BatchEvaluationResults(
        results=relevance_results,
        evaluation_type="relevance"
    )
    
    # Generate summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for name, batch in [
        ("Faithfulness", faithfulness_batch),
        ("Precision", precision_batch),
        ("Relevance", relevance_batch)
    ]:
        stats = evaluate_summary_statistics(batch)
        print(f"{name}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}\n")
    
    # Filter results by metadata
    print("\n=== FILTERED RESULTS (Science Category) ===")
    science_results = filter_results(
        faithfulness_batch,
        metadata_filters={"category": "science"}
    )
    print(f"Count: {len(science_results.results)}")
    print(f"Average score: {science_results.average_score:.2f}")
    
    # Combine metrics into a single DataFrame
    print("\n=== COMBINED METRICS ===")
    metrics_results = {
        "faithfulness": faithfulness_batch,
        "precision": precision_batch,
        "relevance": relevance_batch
    }
    combined_df = combine_metrics(metrics_results)
    print(combined_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    faithfulness_batch.save_csv(f"faithfulness_results_{timestamp}.csv")
    faithfulness_batch.save_json(f"faithfulness_results_{timestamp}.json")
    
    # Plot results
    try:
        plot_path = plot_metric_comparison(metrics_results)
        print(f"\nMetric comparison plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error creating plot: {e}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())