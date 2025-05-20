from __future__ import annotations

import logging
from typing import Any, Optional
from pydantic import BaseModel, ValidationInfo, field_validator, Field
from instructor import Instructor, AsyncInstructor
from textwrap import dedent

logger = logging.getLogger(__name__)


class ContextValidationMixin:
    """Mixin class that ensures the integrity of chunk references in RAG evaluations by validating that all chunk IDs correspond to actual context chunks. This validation is crucial for maintaining data consistency and preventing errors in evaluation metrics like precision and recall."""
    
    @field_validator('graded_chunks')
    @classmethod
    def validate_chunks_against_context(cls, chunks: list[Any], info: ValidationInfo) -> list[Any]:
        """Validate and process chunk IDs against context chunks.
        
        This validator ensures data integrity by:
        1. Collecting any chunk IDs that don't exist in the context and raising an error.
        2. Adding missing context chunks with a score of 0 and issuing a warning.
        3. Maintaining the original order of chunks.
        
        Args:
            chunks: List of chunks with IDs to validate
            info: ValidationInfo containing context. Expected to be a dict with a key 'context'
                  whose value is the list of actual context items.
            
        Returns:
            List of valid chunks, with missing context chunks added (score=0)
            
        Raises:
            ValueError: If any chunk IDs don't exist in the context or if the validation
                        context in `info.context` is missing or malformed.
        """
        if not chunks:
            return []
            
        if not info.context or not isinstance(info.context, dict) or 'context' not in info.context:
            raise ValueError(
                "ContextValidationMixin: Validation context (info.context) must be a dictionary "
                "containing a 'context' key (list of actual context items). "
                "Provide it via model_validate(..., context={'context': ...})."
            )
        
        actual_context_items = info.context['context']
        if not isinstance(actual_context_items, list):
             raise ValueError(
                "ContextValidationMixin: The value of 'context' in validation context (info.context) "
                "must be a list of context items."
            )
            
        # Get all valid chunk IDs from context
        valid_ids = set(range(len(actual_context_items)))
        
        # Track invalid IDs and process chunks
        invalid_ids = set()
        valid_chunks = []
        processed_ids = set()
        
        # First pass: collect valid chunks and track invalid IDs
        for chunk in chunks:
            if not hasattr(chunk, 'id_chunk'):
                logger.warning(f"Chunk `{chunk}` has no `id_chunk` attribute")
                continue
                
            if chunk.id_chunk not in valid_ids:
                invalid_ids.add(chunk.id_chunk)
            else:
                valid_chunks.append(chunk)
                processed_ids.add(chunk.id_chunk)
        
        # Raise error if any invalid IDs were found
        if invalid_ids:
            raise ValueError(f"Found chunk IDs that don't exist in context: {invalid_ids}")
            
        # Second pass: add missing context chunks with score=0
        missing_chunk_ids_added = []
        for chunk_id in valid_ids:
            if chunk_id not in processed_ids:
                valid_chunks.append(ChunkScore(id_chunk=chunk_id, score=0.0))
                missing_chunk_ids_added.append(chunk_id)
        
        if missing_chunk_ids_added:
            logger.warning(
                f"Missing chunk IDs {missing_chunk_ids_added} were not in graded_chunks. "
                f"They have been added with a score of 0.0."
            )
                
        # Sort chunks by ID to maintain consistent order
        valid_chunks.sort(key=lambda x: x.id_chunk)
            
        return valid_chunks

class ChunkScore(BaseModel):
    id_chunk: int
    score: float = Field(ge=0.0, le=1.0, description="Score from 0-1 indicating the precision of the chunk, lower is worse")

class ChunkBinaryScore(BaseModel):
    id_chunk: int
    score: bool = Field(description="Whether the chunk is passed or failed")

class ChunkGraded(BaseModel, ContextValidationMixin):
    graded_chunks: list[ChunkScore]

    @property 
    def avg_score(self) -> float:
        return sum(chunk.score for chunk in self.graded_chunks) / len(self.graded_chunks)

class ChunkGradedBinary(BaseModel, ContextValidationMixin):
    graded_chunks: list[ChunkBinaryScore]

    @property 
    def avg_score(self) -> float:
        return sum(chunk.score for chunk in self.graded_chunks) / len(self.graded_chunks)

class ContextEvaluation(BaseModel):
    """Base class for context-based evaluations that handles common patterns
    including grading question, and optional answers against a context that is enumerated   
    with an id.
    
    This class is designed to be used as a base class for specific evaluation classes.
    It provides a common interface for evaluating questions and answers against a context.
    """
    
    prompt: str
    examples: list[Any] | None = None
    response_model: type[BaseModel]
    chunk_template: str = dedent("""
        <evaluation>
            {% if examples is not none %}
            <examples>
                {% for example in examples %}
                <example>
                    {{ example }}
                </example>
                {% endfor %}
            {% endif %}
            <question>{{ question }}</question>
            {% if answer is not none %}
            <answer>{{ answer }}</answer>
            {% endif %}
            <context>
                {% for chunk in context %}
                <chunk id="{{ chunk.id }}">
                    {{ chunk.chunk }}
                </chunk>
                {% endfor %}
            </context>
        </evaluation>
    """)
    
    def grade(
        self,
        question: str,
        answer: str | None,
        context: list[Any],
        client: Instructor,
    ) -> BaseModel:
        """Run an evaluation of a question and optional answer against provided context chunks.

        This method evaluates how well a question and its answer (if provided) relate to the given
        context chunks. It uses a template-based approach to structure the evaluation request.

        Args:
            question (str): The question being evaluated.
            answer (Optional[str]): The answer to evaluate, if available. Can be None.
            context (List[Any]): List of context chunks to evaluate against. Each chunk will be
                enumerated and included in the evaluation template.
            response_model (type[T]): A Pydantic model class that defines the structure of the
                evaluation response. Must inherit from BaseModelWithScore.
            client (Instructor): An initialized Instructor client instance used to make the
                evaluation request.

        Returns:
            T: An instance of the response_model containing the structured evaluation results.
            The response will include scores and any other fields defined in the response model.

        Note:
            The evaluation uses the class's prompt and chunk_template to structure the request.
            The context chunks are automatically enumerated and included in the template.
        """
        response = client.create(
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.chunk_template}
            ],
            response_model=self.response_model,
            context={
                "question": question,
                "answer": answer,
                "context": [{"id": i, "chunk": chunk} for i, chunk in enumerate(context)]
            }
        )
        return response
        
    async def agrade(
        self,
        question: str,
        answer: str | None,
        context: list[Any],
        client: AsyncInstructor,
    ) -> BaseModel:
        """Run an evaluation of a question and optional answer against provided context chunks.

        This method evaluates how well a question and its answer (if provided) relate to the given
        context chunks. It uses a template-based approach to structure the evaluation request.

        Args:   
            question (str): The question being evaluated.   
            answer (Optional[str]): The answer to evaluate, if available. Can be None.
            context (List[Any]): List of context chunks to evaluate against. Each chunk will be
                enumerated and included in the evaluation template.
            client (AsyncInstructor): An initialized AsyncInstructor client instance used to make the
                evaluation request. 

        Returns:
            T: An instance of the response_model containing the structured evaluation results.
            The response will include scores and any other fields defined in the response model.
        """
        response = await client.create(
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.chunk_template}
            ],
            response_model=self.response_model,
            context={
                "question": question,
                "answer": answer,
                "context": [{"id": i, "chunk": chunk} for i, chunk in enumerate(context)]
            }
        )
        return response