# RAG Evaluation Best Practices

This document provides guidance on effectively using the RAG Evals framework for Retrieval Augmented Generation (RAG) systems.

## Core Evaluation Approach

When evaluating RAG systems, we recommend these fundamental principles:

1. **Evaluate Multiple Dimensions**: Assess both the retrieval and generation components
2. **Use Independent Metrics**: Apply separate metrics for each aspect of performance
3. **Contextual Assessment**: Evaluate answers within the context of the specific question and retrieved information
4. **Structured Evaluation**: Break down complex assessments into atomic verification tasks

## Metric-Specific Best Practices

### Faithfulness Evaluation

- **Atomic Claims**: Break answers into individual, verifiable statements
- **Evidence Citation**: Require specific attribution to context chunks for each claim
- **Strict Criteria**: All aspects of a claim must be verifiable in the context
- **Tracking Support**: Keep detailed records of which context chunks support which statements

```python
# Example: Detailed statement analysis
for statement in faithfulness_result.statements:
    print(f"- {statement.statement}")
    print(f"  Supported: {statement.is_supported}")
    if statement.is_supported:
        print(f"  Supporting chunks: {statement.supporting_chunk_ids}")
        # Examine these chunks to understand the evidence base
```

### Context Precision Evaluation

- **Independent Chunk Assessment**: Evaluate each context chunk individually
- **Focus on Relevance**: Determine if the chunk's information is relevant to the original question
- **Binary Judgment**: Make a clear yes/no decision about chunk relevance
- **Analysis of Irrelevant Chunks**: Examine chunks marked as irrelevant to improve retrieval

```python
# Example: Analyzing irrelevant chunks to improve retrieval
irrelevant_chunks = [chunk for chunk in precision_result.graded_chunks if not chunk.score]
print(f"Number of irrelevant chunks: {len(irrelevant_chunks)}")
print(f"Proportion of relevant chunks: {precision_result.avg_score:.2f}")

# Examine why these chunks weren't relevant
for chunk_score in irrelevant_chunks:
    chunk_id = chunk_score.id_chunk
    chunk_content = context[chunk_id]
    print(f"Irrelevant chunk {chunk_id}: {chunk_content[:100]}...")
```

## Implementation Recommendations

### Model Selection

- **Evaluation Model Quality**: Use the most capable model available for evaluation
- **Model Separation**: Consider using different models for generation vs. evaluation
- **Capability Match**: Ensure the evaluation model can handle the complexity of your evaluation tasks

### Context Handling

- **Chunk Boundaries**: Pay attention to how information is split across chunks
- **Include IDs**: Always track chunk IDs to maintain traceability
- **Consistent Chunking**: Use a consistent chunking strategy across your evaluation dataset

### Scoring Methodology

- **Binary + Nuance**: Combine binary decisions with nuanced assessments
- **Aggregation**: Consider both per-example and dataset-level metrics
- **Multiple Runs**: For non-deterministic evaluations, consider averaging across multiple runs

## Evaluation Workflow

1. **Prepare Evaluation Data**: Create a diverse set of test questions
2. **Run Baseline Evaluations**: Establish performance benchmarks
3. **Analyze Failures**: Identify patterns in low-scoring examples
4. **Implement Improvements**: Based on failure analysis
5. **Re-evaluate**: Measure improvement against the baseline

## Common Pitfalls to Avoid

- **Overly Generous Evaluation**: Maintain strict standards for what counts as "supported" or "relevant"
- **Context Leakage**: Ensure evaluation doesn't use information outside the provided context
- **Cherry-picking Examples**: Use a representative set of test cases
- **Misleading Aggregation**: Report both average scores and score distributions

## Advanced Evaluation Techniques

- **Ablation Studies**: Remove components to measure their impact
- **Comparative Evaluation**: Test multiple RAG configurations side-by-side
- **Human-in-the-loop Calibration**: Periodically compare LLM evaluations with human judgments
- **Multi-metric Scoring**: Develop composite scores that combine multiple metrics

## Distinguishing Related Metrics

Understanding the relationships between different metrics is crucial:

1. **Context Precision vs. Chunk Utility**:
   - **Context Precision**: Measures if a chunk is relevant to the question
   - **Chunk Utility**: Measures if a chunk was actually used in the answer
   
2. **Faithfulness vs. Answer Relevancy**:
   - **Faithfulness**: Measures if the answer is supported by the context
   - **Answer Relevancy**: Measures if the answer addresses the question

By following these best practices, you can develop a robust evaluation framework that helps identify and address issues in your RAG system, leading to consistent improvements in performance.