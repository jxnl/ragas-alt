# Understanding RAG Evaluation Metrics

When we build Retrieval Augmented Generation (RAG) systems, it's important to measure how well they are working. This helps us improve them. Here are some key metrics we use to evaluate different parts of the RAG pipeline.

## 1. Answer Relevancy

*   **What it measures:** How well does the system's final **answer** address the original **user's question**?
*   **Focus:** The relationship is (Generated Answer → User's Question).
*   **Why it's important:** We want answers that are on-topic and directly attempt to solve what the user asked. An answer might be factually correct but irrelevant to the specific question.
*   **Example:**
    *   **User's Question:** "What is the capital of France, and when was the Eiffel Tower completed?"
    *   **Generated Answer (High Relevancy):** "The capital of France is Paris. The Eiffel Tower was completed on March 31, 1889."
        *   *Explanation:* This answer directly addresses both parts of the user's question.
    *   **Generated Answer (Medium Relevancy):** "Paris is the capital of France, known for its rich history and landmarks like the Louvre."
        *   *Explanation:* This answer addresses the first part of the question but misses the second part about the Eiffel Tower.
    *   **Generated Answer (Low Relevancy):** "France is a country in Western Europe, famous for its cuisine and wine."
        *   *Explanation:* This answer is about France but does not provide the capital or information about the Eiffel Tower.

## 2. Faithfulness (also known as Answer Recall or Groundedness)

*   **What it measures:** Is the system's **generated answer** factually supported by the **retrieved context**? It primarily checks if the answer invents information (hallucinates) or contradicts the information found in the retrieved context.
*   **Focus:** The relationship is (Generated Answer → Retrieved Context).
*   **Why it's important:** We need answers that are truthful and based on the information the system looked up. High faithfulness means the system is not making things up.
*   **Example:**
    *   **User's Question:** "Who wrote the play 'Hamlet'?"
    *   **Retrieved Context:** ["William Shakespeare was an English playwright active in the late 16th and early 17th centuries. His famous tragedies include 'Hamlet', 'Othello', and 'King Lear'."]
    *   **Generated Answer (High Faithfulness):** "The play 'Hamlet' was written by William Shakespeare."
        *   *Explanation:* This statement is directly supported by the retrieved context.
    *   **Generated Answer (Low Faithfulness - Hallucination):** "The play 'Hamlet' was written by Charles Dickens."
        *   *Explanation:* This statement is not supported by the retrieved context (and is factually incorrect).
    *   **Generated Answer (Low Faithfulness - Contradiction):** "Based on the context, 'Hamlet' is not one of William Shakespeare's famous tragedies."
        *   *Explanation:* This statement contradicts the information present in the retrieved context.

## 3. Context Precision (Chunk Utility for Answer)

*   **This is what our current `ChunkPrecision` scorer in `score_precision.py` aims to measure.**
*   **What it measures:** For each individual **retrieved chunk** of context, how useful or essential was it for constructing the system's **actual generated answer**?
*   **Focus:** The relationship is (Individual Retrieved Chunk → Generated Answer).
*   **Why it's important:** This metric helps identify if the retriever is fetching chunks that the generator part of the RAG system *actually uses*. If many chunks are retrieved but contribute nothing to the final answer, the retrieval might be inefficient or the generator might be ignoring useful information.
*   **Example:**
    *   **User's Question:** "What are the key benefits of regular exercise?"
    *   **Retrieved Context:**
        *   **Chunk 1:** "Regular physical activity can improve your muscle strength and boost your endurance. Exercise delivers oxygen and nutrients to your tissues and helps your cardiovascular system work more efficiently."
        *   **Chunk 2:** "Being active has been shown to have many health benefits, both physically and mentally. It may even help you live longer."
        *   **Chunk 3:** "The history of athletic competitions dates back to ancient Greece, where games were held in Olympia."
    *   **Generated Answer:** "Regular exercise improves muscle strength, boosts endurance, and helps the cardiovascular system work better. It offers physical and mental health benefits."
    *   **Evaluation for Context Precision:**
        *   **Chunk 1:** High Precision. Information about muscle strength, endurance, and cardiovascular health was directly used in the answer.
        *   **Chunk 2:** Medium to High Precision. The general statement about physical and mental health benefits was used.
        *   **Chunk 3:** Low Precision. This chunk, while about athletics, was not used to form the answer about the *benefits* of exercise.

## 4. Context Relevancy (Chunk Relevance to Question)

*   **This is a common industry metric, different from our current `ChunkPrecision` defined above.**
*   **What it measures:** For each individual **retrieved chunk**, how relevant is its content to the **user's original question**, regardless of whether that chunk was actually used in the final generated answer?
*   **Focus:** The relationship is (Individual Retrieved Chunk → User's Question).
*   **Why it's important:** This metric assesses the quality of the retriever itself. A good retriever should find chunks that are on-topic for the user's query. If Context Relevancy is low, the retriever might be pulling in irrelevant information, even if the generator manages to ignore it.
*   **Example (using a similar scenario as Context Precision for comparison):**
    *   **User's Question:** "What are the key benefits of regular exercise?"
    *   **Retrieved Context:**
        *   **Chunk 1:** "Regular physical activity can improve your muscle strength and boost your endurance. Exercise delivers oxygen and nutrients to your tissues and helps your cardiovascular system work more efficiently."
        *   **Chunk 2:** "Aerobic exercise, like running or swimming, is particularly good for heart health. It's recommended to get at least 150 minutes of moderate aerobic exercise per week."
        *   **Chunk 3:** "The history of athletic competitions dates back to ancient Greece, where games were held in Olympia."
    *   **Generated Answer (example):** "Regular exercise improves muscle strength and boosts endurance." (Let's assume the generator only used Chunk 1 for this answer).
    *   **Evaluation for Context Relevancy:**
        *   **Chunk 1:** High Relevancy. Directly discusses benefits of exercise.
        *   **Chunk 2:** High Relevancy. Discusses types of exercise beneficial for health, which is relevant to the overall question about benefits.
        *   **Chunk 3:** Low Relevancy. While related to "athletics," its historical focus is not directly relevant to the *benefits* of exercise for an individual.
    *   **Nuance with Context Precision:**
        *   In this scenario, if the answer only used Chunk 1:
            *   **Chunk 1:** High Context Precision (used), High Context Relevancy (relevant to Q).
            *   **Chunk 2:** Low Context Precision (not used), High Context Relevancy (relevant to Q). This indicates the retriever found a good chunk, but the generator didn't use it.
            *   **Chunk 3:** Low Context Precision (not used), Low Context Relevancy (not relevant to Q). This indicates the retriever fetched an off-topic chunk.

## 5. Context Recall (Retriever's ability to fetch info for a Ground Truth answer)

*   **What it measures:** Did the set of all **retrieved context chunks** together contain all the necessary information needed to construct an ideal, "ground truth" answer to the **user's question**?
*   **Focus:** The relationship is (Entire Set of Retrieved Context → Ground Truth Answer).
*   **Why it's important:** This tells us if the retriever is finding *enough* of the right information from the knowledge base to comprehensively answer the question. A low score means the retriever is missing key pieces of context that *should* have been found.
*   **Challenge:** This metric typically requires a human-verified "gold standard" or ground truth answer for comparison, which can be time-consuming and resource-intensive to create and maintain for many questions.
*   **Example:**
    *   **User's Question:** "What are the main components and benefits of the RAGAS evaluation framework?"
    *   **Ground Truth Answer (Ideal Answer):** "RAGAS evaluates RAG pipelines using four core metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall. Its benefits include its ability to leverage LLMs for evaluation and that some of its metrics can operate without human-annotated ground truths."
    *   **Retrieved Context (Scenario 1 - High Context Recall):**
        *   **Chunk A:** "The RAGAS framework assesses Retrieval Augmented Generation systems. Its key metrics are Faithfulness (factual accuracy against context) and Answer Relevancy (answer's pertinence to the query)."
        *   **Chunk B:** "RAGAS also includes Context Precision, which evaluates the signal-to-noise ratio of retrieved chunks, and Context Recall, measuring if retrieved documents cover all aspects needed for the answer, often based on a ground truth statement."
        *   **Chunk C:** "A major advantage of RAGAS is its use of large language models (LLMs) as evaluators, which allows for nuanced assessment. For several metrics, RAGAS does not require pre-existing ground truth labels."
    *   **Retrieved Context (Scenario 2 - Low Context Recall):**
        *   **Chunk X:** "RAGAS helps evaluate RAG systems. One important metric is Faithfulness, which is about the answer's factual grounding in the provided context."
        *   **Chunk Y:** "Answer Relevancy is another metric used by RAGAS to see if the answer is on-topic for the question."
        *   *(In this scenario, information about Context Precision, Context Recall, the use of LLMs as evaluators, and the benefit of not always needing ground truths is missing from the retrieved context.)*
    *   **Evaluation:**
        *   **Scenario 1 (High Context Recall):** The combination of Chunks A, B, and C covers all the main components and benefits mentioned in the Ground Truth Answer.
        *   **Scenario 2 (Low Context Recall):** Chunks X and Y only cover two metrics and miss other key components and benefits. The retrieved context is insufficient to construct the full Ground Truth Answer.

By understanding and using these metrics, we can better diagnose issues in our RAG systems, whether they lie in the retriever's ability to find good information or the generator's ability to use that information effectively and truthfully. 