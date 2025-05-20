# Chunk Utility Evaluation

Chunk Utility is an important metric for RAG evaluation that focuses on how effectively each retrieved context chunk contributes to the generated answer.

## What Chunk Utility Measures

- **Definition**: Chunk Utility measures how useful each **retrieved context chunk** was for generating the **answer** to the user's question. 
- **Focus**: The relationship is (Retrieved Context → Question → Generated Answer).
- **Purpose**: To evaluate the utility of individual context chunks and identify which were most and least helpful.

## How It Works

The Chunk Utility evaluator:

1. Examines each context chunk to assess its utility for answering the specific question
2. Assigns a utility score (0.0-1.0) to each chunk based on how much it contributed to the answer
3. Provides a justification for each score
4. Calculates an overall utility score based on the average of individual chunk scores

## Implementation Details

The Chunk Utility implementation follows this approach:

```python
from rag_evals import ChunkUtility

utility_result = await ChunkUtility.agrade(
    question=question,
    answer=answer,
    context=context,
    client=client
)
```

### Response Model

The `ChunkUtilityResult` class contains:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ChunkUtilityScore(BaseModel):
    id_chunk: int = Field(description="The ID of the context chunk being evaluated.")
    utility_score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Score from 0-1 indicating how useful this chunk was for generating the answer."
    )
    justification: str = Field(description="Explanation of why this chunk received its utility score.")

class ChunkUtilityResult(BaseModel):
    evaluated_chunks: List[ChunkUtilityScore] = Field(
        description="Utility evaluation of each context chunk."
    )
    
    @property
    def score(self) -> float:
        """
        Calculate the overall utility score as the average of individual chunk scores.
        """
        if not self.evaluated_chunks:
            return 0.0
        
        return sum(chunk.utility_score for chunk in self.evaluated_chunks) / len(self.evaluated_chunks)
    
    @property
    def most_useful_chunk(self) -> Optional[dict]:
        """
        Returns the most useful chunk and its utility score.
        """
        if not self.evaluated_chunks:
            return None
        
        most_useful = max(self.evaluated_chunks, key=lambda x: x.utility_score)
        return {
            "chunk_id": most_useful.id_chunk,
            "utility_score": most_useful.utility_score,
            "justification": most_useful.justification
        }
    
    @property
    def least_useful_chunk(self) -> Optional[dict]:
        """
        Returns the least useful chunk and its utility score.
        """
        if not self.evaluated_chunks:
            return None
        
        least_useful = min(self.evaluated_chunks, key=lambda x: x.utility_score)
        return {
            "chunk_id": least_useful.id_chunk,
            "utility_score": least_useful.utility_score,
            "justification": least_useful.justification
        }
```

### Utility Score Scale

The utility score follows this general scale:

- **1.0 (Crucial)**: Essential information without which the answer would be impossible or significantly incomplete
- **0.8 (Very Useful)**: Important details that significantly enhance the answer's quality
- **0.6 (Moderately Useful)**: Some relevant information that contributes to the answer
- **0.4 (Somewhat Useful)**: Minimal relevance but provides some context or background
- **0.2 (Barely Useful)**: Tangentially related but not particularly helpful
- **0.0 (Not Useful)**: Completely irrelevant or contains misinformation

### Example Output

```python
# Result example
utility_result = ChunkUtilityResult(
    evaluated_chunks=[
        ChunkUtilityScore(
            id_chunk=0,
            utility_score=0.8,
            justification="This chunk provided key information about weight loss benefits that was central to the answer."
        ),
        ChunkUtilityScore(
            id_chunk=1,
            utility_score=0.4,
            justification="This chunk offered some context but only minimal information was used in the answer."
        ),
        ChunkUtilityScore(
            id_chunk=2,
            utility_score=0.0,
            justification="This chunk was completely irrelevant to the question and not used in the answer."
        )
    ]
)

# Utility score: 0.4 (calculated via utility_result.score)
```

## Customizing the Evaluation

You can customize the utility evaluation prompt:

```python
from rag_evals import ChunkUtility
from rag_evals import base
from rag_evals import ChunkUtilityResult

# Access the original prompt
original_prompt = ChunkUtility.prompt

# Create a customized evaluator with a modified prompt
CustomUtility = base.ContextEvaluation(
    prompt="Your custom prompt here...",
    response_model=ChunkUtilityResult
)
```

## Considerations When Using Chunk Utility

- **Utility vs. Relevance**: A chunk can be relevant to the topic but not useful for the specific question.
- **Scoring Consistency**: Consider how consistently the utility scale is applied across evaluations.
- **Misinformation**: Chunks that introduce errors should receive very low utility scores, even if they appear to be on-topic.

## Best Practices

- Use Chunk Utility to identify which types of context are most helpful for different questions
- Review the justifications to understand why chunks were or weren't useful
- Use the `most_useful_chunk` and `least_useful_chunk` properties to quickly identify outliers
- Consider this metric alongside others like Faithfulness and Context Recall for a complete evaluation

For implementation examples, see the [usage examples](../usage/examples.md).