from __future__ import annotations
import logging
from .. import base

logger = logging.getLogger(__name__)

ChunkPrecision = base.ContextEvaluation(
    prompt = """
    You are an expert evaluator assessing if a specific retrieved context chunk is relevant to the original question.

    ## Your Task:
    Given a user's **question** and a single **retrieved context chunk**, determine if the information in this specific chunk is relevant to answering the original question, regardless of whether it was used in the final answer.

    - Consider if the chunk contains information that would help answer the question, either directly or indirectly.
    - The chunk is considered relevant even if only a small part of it contains pertinent information.
    - Focus solely on the relevance of the chunk to the question, not on whether it was actually used in the answer.
    - A chunk can be topically relevant even if it doesn't contain the exact information needed to answer the question completely.
    """, 
    response_model = base.ChunkGradedBinary
)