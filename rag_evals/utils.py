from __future__ import annotations

import json
import logging
import asyncio
from typing import List, Dict, Any, TypeVar, Generic, Optional, Union, Callable
from pathlib import Path
import pandas as pd
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)
S = TypeVar('S', bound=BaseModel)

class EvaluationResult(BaseModel, Generic[T]):
    """Container for evaluation results with metadata."""
    question: str
    answer: Optional[str] = None
    context: List[str]
    result: T
    metadata: Dict[str, Any] = {}
    
    @property
    def score(self) -> float:
        """Extract the score from the result model."""
        if hasattr(self.result, "score"):
            return self.result.score
        elif hasattr(self.result, "overall_score"):
            return self.result.overall_score
        else:
            logger.warning(f"No score attribute found in result model {type(self.result).__name__}")
            return 0.0

class BatchEvaluationResults(BaseModel, Generic[T]):
    """Container for a batch of evaluation results."""
    results: List[EvaluationResult[T]]
    evaluation_type: str
    timestamp: datetime = datetime.now()
    
    @property
    def scores(self) -> List[float]:
        """Extract scores from all results."""
        return [r.score for r in self.results]
    
    @property
    def average_score(self) -> float:
        """Calculate the average score across all results."""
        if not self.results:
            return 0.0
        return sum(self.scores) / len(self.results)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        data = []
        for r in self.results:
            item = {
                "question": r.question,
                "score": r.score,
                **r.metadata
            }
            
            # Add any additional score attributes from the result model
            if hasattr(r.result, "__dict__"):
                for key, value in r.result.__dict__.items():
                    if key != "score" and isinstance(value, (int, float)) and key not in item:
                        item[f"result_{key}"] = value
            
            data.append(item)
        
        return pd.DataFrame(data)
    
    def save_csv(self, path: Union[str, Path]) -> Path:
        """Save results to a CSV file."""
        df = self.to_dataframe()
        path = Path(path)
        df.to_csv(path, index=False)
        return path
    
    def save_json(self, path: Union[str, Path]) -> Path:
        """Save complete results to a JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            # Convert to dict and handle datetime serialization
            json_data = {
                "evaluation_type": self.evaluation_type,
                "timestamp": self.timestamp.isoformat(),
                "results": [r.model_dump() for r in self.results]
            }
            json.dump(json_data, f, indent=2)
        return path

async def batch_evaluate(
    evaluation_func: Callable,
    data: List[Dict[str, Any]],
    batch_size: int = 10,
    delay_seconds: float = 0.5
) -> List[Any]:
    """
    Process a large dataset in batches with throttling.
    
    Args:
        evaluation_func: Async function to evaluate each item
        data: List of data items to evaluate
        batch_size: Number of items to process in each batch
        delay_seconds: Delay between batches in seconds
        
    Returns:
        List of evaluation results
    """
    all_results = []
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(data)+batch_size-1)//batch_size} ({len(batch)} items)")
        
        # Process this batch
        batch_tasks = [evaluation_func(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle exceptions
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error evaluating item {i+j}: {result}")
                # Replace exception with None to maintain result order
                batch_results[j] = None
        
        # Filter out None results
        batch_results = [r for r in batch_results if r is not None]
        all_results.extend(batch_results)
        
        # Wait before processing next batch (to avoid rate limits)
        if i + batch_size < len(data):
            await asyncio.sleep(delay_seconds)
            
    return all_results

def combine_metrics(results_dict: Dict[str, BatchEvaluationResults]) -> pd.DataFrame:
    """
    Combine results from multiple metrics into a single DataFrame.
    
    Args:
        results_dict: Dictionary mapping metric names to BatchEvaluationResults
        
    Returns:
        DataFrame with combined metrics
    """
    if not results_dict:
        return pd.DataFrame()
    
    # Extract first metric's dataframe as base
    first_metric = next(iter(results_dict.values()))
    combined_df = first_metric.to_dataframe()
    combined_df.rename(columns={"score": f"{first_metric.evaluation_type}_score"}, inplace=True)
    
    # Add scores from other metrics
    for metric_name, results in results_dict.items():
        if metric_name == first_metric.evaluation_type:
            continue
            
        df = results.to_dataframe()
        # Only keep the score column to avoid duplication
        score_col = f"{metric_name}_score"
        combined_df[score_col] = df["score"]
    
    return combined_df

def evaluate_summary_statistics(batch_results: BatchEvaluationResults) -> Dict[str, Any]:
    """
    Calculate summary statistics for a batch of evaluation results.
    
    Args:
        batch_results: Batch evaluation results
        
    Returns:
        Dictionary of summary statistics
    """
    scores = batch_results.scores
    if not scores:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "std": 0
        }
    
    df = pd.DataFrame({"score": scores})
    return {
        "count": len(scores),
        "mean": df["score"].mean(),
        "median": df["score"].median(),
        "min": df["score"].min(),
        "max": df["score"].max(),
        "std": df["score"].std()
    }

def filter_results(
    batch_results: BatchEvaluationResults,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    metadata_filters: Optional[Dict[str, Any]] = None
) -> BatchEvaluationResults:
    """
    Filter results based on score range and metadata.
    
    Args:
        batch_results: Batch evaluation results
        min_score: Minimum score threshold (inclusive)
        max_score: Maximum score threshold (inclusive)
        metadata_filters: Dictionary of metadata keys and values to filter by
        
    Returns:
        Filtered batch results
    """
    filtered_results = []
    
    for result in batch_results.results:
        # Check score against min/max thresholds
        if min_score is not None and result.score < min_score:
            continue
        if max_score is not None and result.score > max_score:
            continue
            
        # Check metadata filters
        if metadata_filters:
            skip = False
            for key, value in metadata_filters.items():
                if key not in result.metadata or result.metadata[key] != value:
                    skip = True
                    break
            if skip:
                continue
                
        filtered_results.append(result)
    
    return BatchEvaluationResults(
        results=filtered_results,
        evaluation_type=batch_results.evaluation_type,
        timestamp=batch_results.timestamp
    )