# Batch Evaluation

This guide demonstrates how to efficiently evaluate multiple RAG system outputs in batch.

## Why Batch Evaluation?

Batch evaluation offers several advantages:

1. **Efficiency**: Process multiple evaluations in parallel
2. **Consistency**: Apply the same evaluation criteria across all samples
3. **Aggregation**: Analyze trends and patterns across a dataset
4. **Benchmarking**: Compare performance across different system configurations

## Basic Batch Evaluation

The core concept of batch evaluation is to process multiple question-answer-context sets in parallel. Here's a simple example:

```python
import asyncio
import instructor
from rag_evals import Faithfulness

# Initialize client
client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

# Sample data
evaluation_data = [
    {
        "question": "What are the benefits of exercise?",
        "answer": "Regular exercise improves cardiovascular health and increases strength.",
        "context": [
            "Regular physical activity improves heart health and circulation.",
            "Weight training builds muscle strength and increases bone density.",
            "The earliest Olympic games were held in Ancient Greece."
        ]
    },
    # More evaluation items...
]

async def evaluate_batch(data):
    # Create tasks for all items
    tasks = []
    for item in data:
        task = Faithfulness.agrade(
            question=item["question"],
            answer=item["answer"],
            context=item["context"],
            client=client
        )
        tasks.append(task)
    
    # Run all tasks in parallel and return results
    return await asyncio.gather(*tasks)

# Run the batch evaluation
results = asyncio.run(evaluate_batch(evaluation_data))
```

## Comprehensive Batch Evaluation

For a more comprehensive approach, you can:

1. Apply multiple metrics to each item
2. Store detailed results for further analysis
3. Generate summary statistics and visualizations

Here's a more complete example:

```python
import asyncio
import instructor
import pandas as pd
from datetime import datetime
from rag_evals import Faithfulness, ChunkPrecision, AnswerRelevance

# Initialize the async LLM client
client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

async def evaluate_item(item):
    """Evaluate a single item using multiple metrics."""
    question = item["question"]
    answer = item["answer"]
    context = item["context"]
    
    # Run multiple evaluations in parallel
    faithfulness_task = Faithfulness.agrade(
        question=question, answer=answer, context=context, client=client
    )
    precision_task = ChunkPrecision.agrade(
        question=question, answer=None, context=context, client=client
    )
    relevance_task = AnswerRelevance.agrade(
        question=question, answer=answer, context=context, client=client
    )
    
    # Await all tasks
    faithfulness_result, precision_result, relevance_result = await asyncio.gather(
        faithfulness_task, precision_task, relevance_task
    )
    
    # Compile results
    return {
        "id": item.get("id", "unknown"),
        "question": question,
        "faithfulness_score": faithfulness_result.score,
        "precision_score": precision_result.score,
        "relevance_score": relevance_result.overall_score,
        # Store detailed results if needed
        "detailed_results": {
            "faithfulness": faithfulness_result,
            "precision": precision_result,
            "relevance": relevance_result
        }
    }

async def batch_evaluate(data):
    """Run batch evaluation on multiple items in parallel."""
    tasks = [evaluate_item(item) for item in data]
    return await asyncio.gather(*tasks)

def analyze_results(results):
    """Analyze and present evaluation results."""
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            "id": r["id"],
            "question": r["question"],
            "faithfulness": r["faithfulness_score"],
            "precision": r["precision_score"],
            "relevance": r["relevance_score"],
        } 
        for r in results
    ])
    
    # Print summary statistics
    print(f"Total items evaluated: {len(df)}")
    print(f"Average faithfulness: {df['faithfulness'].mean():.2f}")
    print(f"Average precision: {df['precision'].mean():.2f}")
    print(f"Average relevance: {df['relevance'].mean():.2f}")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"evaluation_results_{timestamp}.csv", index=False)
```

## Handling Large Datasets

For very large datasets, you may need to consider:

1. **Batch Size Control**: Process data in smaller batches to avoid rate limits
2. **Throttling**: Add delays between batches to manage API usage
3. **Error Handling**: Deal with potential failures gracefully

Here's an example of batch processing with throttling:

```python
async def batch_evaluate_with_throttling(data, batch_size=10, delay_seconds=1):
    """Process a large dataset in batches with throttling."""
    all_results = []
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        # Process this batch
        batch_results = await asyncio.gather(
            *[evaluate_item(item) for item in batch]
        )
        all_results.extend(batch_results)
        
        # Wait before processing next batch (to avoid rate limits)
        if i + batch_size < len(data):
            await asyncio.sleep(delay_seconds)
            
    return all_results
```

## Complete Example

For a complete working example, see the [batch_evaluation.py](https://github.com/jxnl/rag-evals/blob/main/examples/batch_evaluation.py) script in the examples directory.

The script demonstrates:
- Processing multiple evaluation items in parallel
- Applying multiple metrics to each item
- Aggregating results and generating statistics
- Saving results to CSV for further analysis

## Performance Considerations

When running batch evaluations, consider these performance factors:

1. **API Rate Limits**: Check your LLM provider's rate limits
2. **Memory Usage**: For large datasets, monitor memory consumption
3. **Error Handling**: Implement robust error handling for API failures
4. **Cost Management**: Batch evaluation can quickly consume API tokens, so monitor usage

By following these batch evaluation patterns, you can efficiently evaluate large datasets of RAG system outputs and gain deeper insights into system performance.