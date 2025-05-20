# Understanding RAG Evaluation Metrics

When building Retrieval Augmented Generation (RAG) systems, it's crucial to measure performance across different dimensions. This documentation explains the core metrics implemented in RAG Evals.

## Implemented Metrics

RAG Evals currently implements these key evaluation metrics:

### Faithfulness (Answer Grounding)

Measures whether the generated answer contains only factual claims supported by the retrieved context. It helps detect hallucinations (made-up information).

- **What it measures:** Is the system's **generated answer** factually supported by the **retrieved context**? It checks if the answer invents information (hallucinates) or contradicts the information found in the retrieved context.
- **Focus:** The relationship is (Generated Answer → Retrieved Context).
- **Input**: Question, Answer, Retrieved Context
- **Output**: Statement-level breakdown of which claims are supported, with specific attribution to context chunks

[Learn more about Faithfulness](faithfulness.md)

### Context Precision (Chunk Relevancy)

Evaluates whether each individual retrieved context chunk is relevant to the original question.

- **What it measures:** For each individual **retrieved chunk** of context, how relevant is its content to the **user's original question**, regardless of whether that chunk was actually used in the final generated answer?
- **Focus:** The relationship is (Individual Retrieved Chunk → User's Question).
- **Input**: Question, Answer, Retrieved Context
- **Output**: Binary score for each chunk indicating whether it is relevant to the question

[Learn more about Context Precision](precision.md)

### Answer Relevance

Evaluates how well the generated answer addresses the original question.

- **What it measures:** How well does the **generated answer** address the **user's original question**, regardless of factual accuracy or context support?
- **Focus:** The relationship is (Generated Answer → User's Question).
- **Input**: Question, Answer, Context (though context isn't used in the evaluation)
- **Output**: Multi-dimensional score assessing topical match, completeness, and conciseness

[Learn more about Answer Relevance](relevance.md)

## Understanding the Difference Between Metrics

It's important to understand how these metrics differ from each other:

### Faithfulness vs. Context Precision

- **Faithfulness** starts with the answer and verifies if each statement in it is supported by the context
- **Context Precision** starts with each context chunk and determines if it is relevant to the original question

### Faithfulness vs. Answer Relevance

- **Faithfulness** measures how well the answer is grounded in the context (factual correctness)
- **Answer Relevance** measures how well the answer addresses the question (response appropriateness)

### Context Precision vs. Answer Relevance

- **Context Precision** evaluates the retrieval component by checking chunk relevance to the question
- **Answer Relevance** evaluates the generation component by checking answer responsiveness to the question

## Systematic Decomposition of RAG Evaluations

For a comprehensive understanding of how different evaluation metrics relate to each other and form a complete evaluation framework, see our [Systematic Decomposition of RAG Evaluations](systematic-decomposition.md) guide.

This framework helps you understand the relationships between questions, context chunks, and answers, and how different evaluation metrics assess these relationships.

## Future Metrics

Future releases of RAG Evals plan to include:

- **Context Recall**: Evaluates if the retrieved context contains all information needed for an ideal answer
- **Chunk Utility**: Assesses whether each retrieved chunk was actually used in generating the answer
- **Citation Quality**: Measures how well the system attributes information to sources
- **Response Coherence**: Evaluates the overall structure and flow of generated answers

For more details on best practices when using these metrics, see our [Evaluation Best Practices](../usage/best_practices.md) guide.