import pytest
from rag_evals.base import ChunkScore, ChunkGraded

def test_chunk_graded_validation():
    """Test ChunkGraded model with context validation"""
    # Context with 3 chunks
    context_strings_list = ["Chunk 0", "Chunk 1", "Chunk 2"]
    context_structured_list = [{"id": i, "text": text} for i, text in enumerate(context_strings_list)]
    
    # Valid chunks
    model_data_valid = {
        "graded_chunks": [
            ChunkScore(id_chunk=0, score=0.5),
            ChunkScore(id_chunk=1, score=0.7)
        ]
    }
    validation_context_valid = {'context': context_structured_list}
    valid_graded = ChunkGraded.model_validate(model_data_valid, context=validation_context_valid)

    assert valid_graded.graded_chunks[0].id_chunk == 0
    assert valid_graded.graded_chunks[1].id_chunk == 1
    
    # Note that the third chunk is not included in the model_data_valid, but it is included in the context
    # and thus the score is 0 (default)
    assert valid_graded.graded_chunks[2].score == 0
    
    # Invalid chunks - should raise ValidationError
    with pytest.raises(ValueError):
        model_data_invalid = {
            "graded_chunks": [
                ChunkScore(id_chunk=0, score=0.5),
                ChunkScore(id_chunk=5, score=0.7)  # Invalid ID
            ]
        }
        validation_context_invalid = {'context': context_strings_list}
        ChunkGraded.model_validate(model_data_invalid, context=validation_context_invalid)