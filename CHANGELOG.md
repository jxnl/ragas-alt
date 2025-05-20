# Changelog

All notable changes to RAG Evals will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive API reference documentation for base classes
- Batch evaluation examples and documentation
- Utility functions for processing evaluation results
- Contribution guidelines and changelog

## [0.1.0] - 2023-05-20

### Added
- Initial release of RAG Evals
- Core evaluation infrastructure in `base.py`
- Implementation of three metrics:
  - Faithfulness: Evaluates if answers are factually consistent with context
  - Precision: Evaluates if context chunks are relevant to the question
  - Relevance: Evaluates how well answers address questions
- Basic validation for context chunks
- Example scripts for each metric
- MkDocs documentation setup
- Basic test infrastructure

[Unreleased]: https://github.com/jxnl/rag-evals/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jxnl/rag-evals/releases/tag/v0.1.0