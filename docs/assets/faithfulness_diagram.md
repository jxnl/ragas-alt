```mermaid
graph LR
    Q[Question] --> A[Answer]
    C[Context Chunks] --> A
    A --> |decompose| S1[Statement 1]
    A --> |decompose| S2[Statement 2]
    A --> |decompose| S3[Statement 3]
    S1 --> |evaluate| E1[Supported: Yes\nChunks: [0]]
    S2 --> |evaluate| E2[Supported: Yes\nChunks: [1, 2]]
    S3 --> |evaluate| E3[Supported: No\nChunks: None]
    E1 --> |calculate| F[Faithfulness\nScore: 2/3 = 0.67]
    E2 --> |calculate| F
    E3 --> |calculate| F
    
    style Q fill:#d0e6fa,stroke:#0366d6
    style A fill:#d0e6fa,stroke:#0366d6
    style C fill:#d0e6fa,stroke:#0366d6
    style F fill:#cdfadc,stroke:#28a745
```