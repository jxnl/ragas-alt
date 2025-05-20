# Utility Functions API Reference

This document provides detailed information about the utility functions in RAG Evals for processing, analyzing, and visualizing evaluation results.

## Core Result Classes

### EvaluationResult

`EvaluationResult` is a generic container for a single evaluation result with metadata.

```python
class EvaluationResult(BaseModel, Generic[T]):
    """Container for evaluation results with metadata."""
    question: str
    answer: Optional[str] = None
    context: List[str]
    result: T
    metadata: Dict[str, Any] = {}
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `question` | `str` | The question being evaluated. |
| `answer` | `Optional[str]` | The answer being evaluated. Can be `None` for some evaluations. |
| `context` | `List[str]` | List of context chunks used for evaluation. |
| `result` | `T` | The evaluation result object (generic type). |
| `metadata` | `Dict[str, Any]` | Optional metadata associated with this evaluation. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `score` | `float` | Extracts the score from the result model, looking for `score` or `overall_score` attributes. |

### BatchEvaluationResults

`BatchEvaluationResults` is a container for a batch of evaluation results of the same type.

```python
class BatchEvaluationResults(BaseModel, Generic[T]):
    """Container for a batch of evaluation results."""
    results: List[EvaluationResult[T]]
    evaluation_type: str
    timestamp: datetime = datetime.now()
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `results` | `List[EvaluationResult[T]]` | List of evaluation results. |
| `evaluation_type` | `str` | The type of evaluation (e.g., "faithfulness"). |
| `timestamp` | `datetime` | When the evaluation was performed. Defaults to now. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `scores` | `List[float]` | List of all scores from the results. |
| `average_score` | `float` | The average score across all results. |

#### Methods

##### to_dataframe

```python
def to_dataframe(self) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
```

Converts the batch results to a pandas DataFrame for analysis.

**Returns**:
- A pandas DataFrame with one row per evaluation result.

##### save_csv

```python
def save_csv(self, path: Union[str, Path]) -> Path:
    """Save results to a CSV file."""
```

Saves the results to a CSV file.

**Parameters**:
- `path` (`Union[str, Path]`): Path where the CSV file will be saved.

**Returns**:
- The path to the saved file.

##### save_json

```python
def save_json(self, path: Union[str, Path]) -> Path:
    """Save complete results to a JSON file."""
```

Saves the complete results (including all details) to a JSON file.

**Parameters**:
- `path` (`Union[str, Path]`): Path where the JSON file will be saved.

**Returns**:
- The path to the saved file.

## Batch Processing Functions

### batch_evaluate

```python
async def batch_evaluate(
    evaluation_func: Callable,
    data: List[Dict[str, Any]],
    batch_size: int = 10,
    delay_seconds: float = 0.5
) -> List[Any]:
    """Process a large dataset in batches with throttling."""
```

Process a large dataset in batches with throttling to manage API rate limits.

**Parameters**:
- `evaluation_func` (`Callable`): Async function to evaluate each item.
- `data` (`List[Dict[str, Any]]`): List of data items to evaluate.
- `batch_size` (`int`): Number of items to process in each batch. Default is 10.
- `delay_seconds` (`float`): Delay between batches in seconds. Default is 0.5.

**Returns**:
- List of evaluation results.

## Analysis Functions

### combine_metrics

```python
def combine_metrics(results_dict: Dict[str, BatchEvaluationResults]) -> pd.DataFrame:
    """Combine results from multiple metrics into a single DataFrame."""
```

Combines results from multiple metrics into a single pandas DataFrame for joint analysis.

**Parameters**:
- `results_dict` (`Dict[str, BatchEvaluationResults]`): Dictionary mapping metric names to BatchEvaluationResults.

**Returns**:
- A pandas DataFrame with combined metrics.

### evaluate_summary_statistics

```python
def evaluate_summary_statistics(batch_results: BatchEvaluationResults) -> Dict[str, Any]:
    """Calculate summary statistics for a batch of evaluation results."""
```

Calculates summary statistics (mean, median, min, max, std) for a batch of evaluation results.

**Parameters**:
- `batch_results` (`BatchEvaluationResults`): Batch evaluation results.

**Returns**:
- Dictionary of summary statistics.

### filter_results

```python
def filter_results(
    batch_results: BatchEvaluationResults,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    metadata_filters: Optional[Dict[str, Any]] = None
) -> BatchEvaluationResults:
    """Filter results based on score range and metadata."""
```

Filters evaluation results based on score thresholds and metadata values.

**Parameters**:
- `batch_results` (`BatchEvaluationResults`): Batch evaluation results to filter.
- `min_score` (`Optional[float]`): Minimum score threshold (inclusive). Default is None.
- `max_score` (`Optional[float]`): Maximum score threshold (inclusive). Default is None.
- `metadata_filters` (`Optional[Dict[str, Any]]`): Dictionary of metadata keys and values to filter by. Default is None.

**Returns**:
- Filtered batch results.

## Usage Examples

### Basic Result Processing

```python
from rag_evals.utils import EvaluationResult, BatchEvaluationResults

# Create individual evaluation results
result1 = EvaluationResult(
    question="What is photosynthesis?",
    answer="Photosynthesis is the process...",
    context=["Photosynthesis occurs in...", "The process converts..."],
    result=faithfulness_result,
    metadata={"category": "science", "difficulty": "medium"}
)

result2 = EvaluationResult(
    question="Who was Marie Curie?",
    answer="Marie Curie was...",
    context=["Marie Curie was a physicist...", "She discovered..."],
    result=faithfulness_result2,
    metadata={"category": "history", "difficulty": "medium"}
)

# Create a batch of results
batch_results = BatchEvaluationResults(
    results=[result1, result2],
    evaluation_type="faithfulness"
)

# Get average score
print(f"Average score: {batch_results.average_score}")

# Save to CSV
batch_results.save_csv("faithfulness_results.csv")

# Save to JSON
batch_results.save_json("faithfulness_results.json")
```

### Batch Processing

```python
from rag_evals.utils import batch_evaluate

async def evaluate_single_item(item):
    # Process a single evaluation item
    result = await metric.agrade(
        question=item["question"],
        answer=item["answer"],
        context=item["context"],
        client=client
    )
    return result

# Process a large dataset in batches
results = await batch_evaluate(
    evaluation_func=evaluate_single_item,
    data=large_dataset,
    batch_size=10,
    delay_seconds=1
)
```

### Combining and Analyzing Metrics

```python
from rag_evals.utils import combine_metrics, evaluate_summary_statistics, filter_results

# Combine multiple metrics
metrics_results = {
    "faithfulness": faithfulness_batch,
    "precision": precision_batch,
    "relevance": relevance_batch
}
combined_df = combine_metrics(metrics_results)

# Calculate summary statistics
stats = evaluate_summary_statistics(faithfulness_batch)
print(f"Mean: {stats['mean']}")
print(f"Median: {stats['median']}")
print(f"Range: {stats['min']} - {stats['max']}")

# Filter results by metadata
science_results = filter_results(
    faithfulness_batch,
    metadata_filters={"category": "science"}
)
print(f"Science category average: {science_results.average_score}")
```

For a complete working example, see the [using_utils.py](https://github.com/jxnl/rag-evals/blob/main/examples/using_utils.py) script in the examples directory.