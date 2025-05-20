from __future__ import annotations
import logging
from . import base

logger = logging.getLogger(__name__)

ChunkPrecision = base.ContextEvaluation(
    prompt = """
    You are an expert evaluator assessing if a specific retrieved context chunk was utilized in generating a given answer.

    ## Your Task:
    Given a RAG system's **generated answer** and a single **retrieved context chunk**, determine if the information in this specific chunk was used, wholly or in part, to formulate the generated answer.

    - Consider if the chunk provides any direct or indirect support for statements made in the answer.
    - The chunk is considered utilized even if only a small part of it was relevant to the answer.
    - Focus solely on whether this chunk contributed to the answer, not on the answer's overall correctness or relevance to an original question.
    """
    , 
    response_model = base.ChunkGradedBinary
)