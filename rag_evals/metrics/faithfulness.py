from __future__ import annotations
import logging
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from .. import base

logger = logging.getLogger(__name__)

class StatementEvaluation(BaseModel):
    statement: str = Field(description="An individual claim extracted from the generated answer.")
    is_supported: bool = Field(description="Is this statement supported by the provided context chunks?")
    supporting_chunk_ids: Optional[list[int]] = Field(
        default=None, 
        description="A list of chunk IDs (0-indexed integers) from the provided context that support this statement. Null if not supported or if IDs are not applicable/found."
    )

    @field_validator('supporting_chunk_ids', mode='before')
    @classmethod
    def ensure_list_of_ints(cls, v: Any, info: ValidationInfo) -> Optional[list[int]]:
        # v is the value for supporting_chunk_ids. info contains the validation context.
        
        # If supporting_chunk_ids (v) is None or an empty list, it's considered valid
        # without needing to check against context. Pydantic will handle Optional.
        if v is None or (isinstance(v, list) and not v):
            return v

        # If v is not None and not empty, it should be a list for further processing.
        if not isinstance(v, list):
            # Pydantic will likely catch this when trying to parse into list[int].
            # We pass it through; if it's not a list, Pydantic's own validation for the field type will fail.
            # This validator focuses on content validation assuming a list structure for non-empty 'v'.
            return v
        
        assert info.context is not None, "No context found"

        context_chunks_data = info.context.get('context', {}) # Key is 'context'
        chunk_ids = {chunk['id'] for chunk in context_chunks_data}

        assert len(chunk_ids) > 0, "No context chunks found"
        assert len(v) > 0, "No supporting chunk IDs found"
        
        # Now, 'v' is a non-empty list, and 'context_chunks_data' is a non-empty list.
        # We need to validate each item in 'v'.
        error_messages = []
        for chunk_id in v:
            if chunk_id not in chunk_ids:
                error_messages.append(f"Chunk ID {chunk_id} not found in context")
        
        if error_messages:
            # Join all collected error messages.
            raise ValueError("Validation failed for 'supporting_chunk_ids': " + "; ".join(error_messages))
            
        # If all IDs are valid integers and within range, return 'v'.
        # Pydantic will then complete its validation for Optional[list[int]] using this value.
        return v
    
    @model_validator(mode='after')
    def validate_supporting_chunks(self) -> 'StatementEvaluation':
        """Validate that if a statement is supported, it has supporting chunk IDs.
        
        Returns:
            StatementEvaluation: The validated model instance
            
        Raises:
            ValueError: If is_supported is True but no supporting chunk IDs are provided
        """
        if self.is_supported and not self.supporting_chunk_ids:
            raise ValueError("A supported statement must have at least one supporting chunk ID")
        return self

class FaithfulnessResult(BaseModel):
    statements: list[StatementEvaluation] = Field(description="A list of all statements extracted from the answer and their evaluation.")

    @property
    def overall_faithfulness_score(self) -> float:
        if not self.statements:
            return 0.0
        supported_statements = sum(s.is_supported for s in self.statements)
        return supported_statements / len(self.statements)

Faithfulness = base.ContextEvaluation(
    prompt="""
    You are an expert evaluator tasked with assessing the factual faithfulness of a generated answer to its provided context. Your goal is to break down the answer into individual, verifiable statements and then, for each statement, determine if it is supported by the given context chunks. You must also cite the specific chunk IDs that support each statement.

    ## Input:
    You will be provided with:
    1.  **Generated Answer**: The answer produced by a RAG system.
    2.  **Retrieved Context**: A list of context chunks, each with a 0-indexed integer ID (e.g., <chunk id="0">, <chunk id="1">, etc.).

    ## Your Task:
    1.  **Deconstruct the Answer**: Identify and list all individual factual claims or statements made in the **Generated Answer**.
    2.  **Verify Each Statement**: For each statement, carefully check if it is directly supported by the information present in the **Retrieved Context** chunks.
    3.  **Cite Evidence**: If a statement is supported, you MUST list the 0-indexed integer IDs of all context chunks that provide evidence for it.
    4.  **Provide Reasoning**: For each statement, briefly explain your reasoning for marking it as supported or unsupported, referencing the context or lack thereof.

    ## Output Format:
    Your output MUST be a JSON object that adheres to the following structure:
    {{ "statements": [ {{ "statement": "<The extracted claim from the answer>", "is_supported": <true_or_false>, "reasoning": "<Your brief reasoning>", "supporting_chunk_ids": [<id1>, <id2>, ...] }} , ... ] }}

    - `statement`: The exact claim extracted from the generated answer.
    - `is_supported`: A boolean (true/false) indicating if the statement is supported by the context.
    - `reasoning`: A brief explanation of your decision.
    - `supporting_chunk_ids`: A list of 0-indexed integer chunk IDs that support the statement. If the statement is not supported, or if IDs are not applicable, this can be null or an empty list.

    ## Important Considerations:
    -   Focus SOLELY on faithfulness to the provided context. Do NOT judge the answer's relevance to any original question or its general correctness if not verifiable from the context.
    -   A statement is supported if the context directly states it, or if it can be logically inferred from the context with high confidence.
    -   If multiple chunks support a single statement, list all their IDs.
    -   If a statement is partially supported, mark `is_supported` as false and explain the discrepancy in your reasoning.
    -   Pay close attention to the 0-indexed integer IDs of the context chunks provided in the input when listing `supporting_chunk_ids`.
    """,
    response_model=FaithfulnessResult
)