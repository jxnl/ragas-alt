import pytest
from rag_evals.base import ChunkScore, ChunkGraded

def test_chunk_graded_validation():
    """Test ChunkGraded model with context validation"""
    # Context with 3 chunks
    context_strings_list = ["Chunk 0", "Chunk 1", "Chunk 2"]
    context_structured_list = [{"id": 0, "text": "Chunk 0"}, {"id": 1, "text": "Chunk 1"}, {"id": 2, "text": "Chunk 2"}]
    
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