# Metrics Interaction and Operation

This page provides a detailed explanation of how RAG Evals' metrics operate and interact with each other to provide a comprehensive evaluation of RAG systems.

## Understanding Faithfulness and ChunkPrecision

These two metrics evaluate different aspects of the RAG process, but they work together to provide a complete picture of how well the retrieval and generation components are working.

### Faithfulness: Answer-Context Alignment

![Faithfulness Visualization](../assets/faithfulness_diagram.png)

**Objective**: Evaluate whether the generated answer contains only claims that are supported by the retrieved context.

**Key Characteristics**:
- Focuses on the A|C relationship (Answer given Context)
- Breaks the answer down into factual statements
- Verifies each statement against the context chunks
- Identifies which specific chunks support each statement
- Calculates an overall faithfulness score based on the proportion of supported statements

**Process Flow**:
1. The LLM extracts discrete factual statements from the answer
2. For each statement, it determines if the statement is supported by the context
3. For supported statements, it identifies which specific context chunks provide evidence
4. The overall faithfulness score is calculated as: 
   ```
   score = (number of supported statements) / (total number of statements)
   ```

**Example**:
```python
from rag_evals import Faithfulness
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")

question = "What are the benefits of exercise?"
answer = "Regular exercise improves heart health and builds strength."
context = [
    "Regular physical activity improves cardiovascular health.",
    "Weight training increases muscle strength and bone density.",
    "Exercise can improve mental health and reduce stress levels."
]

result = Faithfulness.grade(
    question=question,
    answer=answer,
    context=context,
    client=client
)

# Statements extracted and evaluated
for stmt in result.statements:
    print(f"Statement: {stmt.statement}")
    print(f"Supported: {stmt.is_supported}")
    print(f"Supporting chunks: {stmt.supporting_chunk_ids}")
    print()

# Overall faithfulness score
print(f"Faithfulness score: {result.score}")
```

**Implementation Details**:
The Faithfulness metric uses a carefully crafted prompt that instructs the LLM to:
1. Deconstruct the answer into individual, verifiable statements
2. Verify each statement against the context
3. Cite specific chunk IDs that support each statement
4. Provide reasoning for why a statement is supported or not

The response is structured using the `FaithfulnessResult` model which contains a list of `StatementEvaluation` objects, each representing an individual statement with its evaluation results.

### ChunkPrecision: Context-Question Relevance

![Precision Visualization](../assets/precision_diagram.png)

**Objective**: Evaluate whether each retrieved context chunk is relevant to the original question.

**Key Characteristics**:
- Focuses on the C|Q relationship (Context given Question)
- Evaluates each context chunk independently
- Uses binary classification (relevant/not relevant)
- Does not require the answer to perform evaluation
- Helps identify irrelevant or low-quality retrieved chunks

**Process Flow**:
1. For each context chunk, the LLM determines if it contains information relevant to the question
2. Relevance is binary - each chunk is either relevant (1) or not relevant (0)
3. The overall precision score is calculated as:
   ```
   score = (number of relevant chunks) / (total number of chunks)
   ```

**Example**:
```python
from rag_evals import ChunkPrecision
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")

question = "What are the benefits of exercise?"
context = [
    "Regular physical activity improves cardiovascular health.",
    "Weight training increases muscle strength and bone density.",
    "The history of the Olympic Games dates back to ancient Greece.",
    "Exercise can improve mental health and reduce stress levels."
]

result = ChunkPrecision.grade(
    question=question,
    answer=None,  # Answer not needed for Precision evaluation
    context=context,
    client=client
)

# Check which chunks were relevant
for i, chunk in enumerate(result.graded_chunks):
    print(f"Chunk {i}: {context[i]}")
    print(f"Relevant: {chunk.score}")
    print()

# Overall precision score
print(f"Precision score: {result.score}")
```

**Implementation Details**:
The ChunkPrecision metric uses a prompt that instructs the LLM to:
1. Consider if the chunk contains information that would help answer the question
2. Mark the chunk as relevant even if only a small part contains pertinent information
3. Focus solely on relevance to the question, not on whether it was used in the answer

The response uses the `ChunkGradedBinary` model, which contains a list of `ChunkBinaryScore` objects, each representing a chunk with a binary score (True/False).

## How Faithfulness and ChunkPrecision Interact

![Metrics Interaction](../assets/metrics_interaction.png)

These metrics complement each other to provide a more complete evaluation of the RAG system:

| Metric | Focus | Evaluates | Requires Answer | Score Type |
|--------|-------|-----------|----------------|------------|
| Faithfulness | Answer-Context consistency | Whether answers are factually grounded in context | Yes | Continuous (0-1) |
| ChunkPrecision | Context-Question relevance | Whether retrieved chunks are relevant to the question | No | Binary, averaged (0-1) |

**Complementary Information**:
- **Faithfulness** tells you if the answer sticks to the facts in the context
- **ChunkPrecision** tells you if the retrieval is bringing in relevant information

**Common Patterns and Insights**:

1. **High Precision, Low Faithfulness**:
   - The system is retrieving relevant chunks but generating answers that aren't supported by them
   - Possible issue: The generation model is hallucinating or ignoring the context

2. **Low Precision, High Faithfulness**:
   - The system is retrieving mostly irrelevant chunks but generating answers that stick to the few relevant parts
   - Possible issue: Inefficient retrieval, but good generation behavior

3. **Low Precision, Low Faithfulness**:
   - Both retrieval and generation are problematic
   - The system needs fundamental improvements in both components

4. **High Precision, High Faithfulness**:
   - Ideal scenario: Good retrieval bringing relevant information and generation using that information faithfully
   - This is the goal for a well-functioning RAG system

## Combined Evaluation

For a holistic evaluation, it's recommended to use both metrics together, along with the AnswerRelevance metric:

```python
import asyncio
from rag_evals import Faithfulness, ChunkPrecision, AnswerRelevance
import instructor

async def evaluate_rag_system(question, answer, context):
    client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)
    
    # Run all evaluations in parallel
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
    faithfulness, precision, relevance = await asyncio.gather(
        faithfulness_task, precision_task, relevance_task
    )
    
    # Calculate comprehensive score (weighted average)
    composite_score = (
        0.4 * faithfulness.score +  # Weight faithfulness more
        0.3 * precision.score +     
        0.3 * relevance.overall_score
    )
    
    return {
        "faithfulness": faithfulness.score,
        "precision": precision.score,
        "relevance": relevance.overall_score,
        "composite_score": composite_score
    }
```

## View as Markdown

To make it easier to feed this documentation to LLMs for further analysis, all documentation pages are available in their raw markdown format. Simply add `.md` to the end of any documentation URL.

For example, to view this page as markdown:
- Web URL: `https://docs.ragevals.com/metrics/interaction`
- Markdown URL: `https://docs.ragevals.com/metrics/interaction.md`

This allows you to easily feed the documentation into LLMs for analysis or to generate custom explanations.