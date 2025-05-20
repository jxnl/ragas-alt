```mermaid
graph LR
    subgraph "RAG System Components"
        Q[Question] --> R[Retrieval]
        R --> C[Context Chunks]
        Q --> G[Generation]
        C --> G
        G --> A[Answer]
    end

    subgraph "Evaluation Metrics"
        C --> |C|Q Relationship| CP[ChunkPrecision]
        Q --> CP
        A --> |A|C Relationship| F[Faithfulness]
        C --> F
        A --> |A|Q Relationship| AR[AnswerRelevance]
        Q --> AR
    end
    
    style Q fill:#d0e6fa,stroke:#0366d6
    style C fill:#d0e6fa,stroke:#0366d6
    style A fill:#d0e6fa,stroke:#0366d6
    style CP fill:#cdfadc,stroke:#28a745
    style F fill:#cdfadc,stroke:#28a745
    style AR fill:#cdfadc,stroke:#28a745
```