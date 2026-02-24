# RAG Chatbot - Domain-Specific AI Chatbot with RAG and Evaluation

A Retrieval-Augmented Generation chatbot that answers user queries based on uploaded domain-specific documents. Built with Gemini, LangChain, FAISS, and evaluated using DeepEval.

## Architecture

```
User Query
    |
    v
[Streamlit UI] --> [Domain Description] --> [Dynamic Prompt Builder (Gemini)]
    |                                              |
    v                                              v
[Document Upload] --> [Chunking] --> [Gemini Embeddings] --> [FAISS Vector Store]
    |
    v
[User Question] --> [Retriever (Similarity Search)] --> [Top-K Chunks]
                                                            |
                                                            v
                                                    [Generator (Gemini)]
                                                            |
                                                            v
                                                    [JSON Response: Answer + Citations]
```

### Components

- **Document Loader**: Extracts text from TXT, PDF, and DOCX files.
- **Retriever**: Uses FAISS vector store with Gemini embeddings for semantic similarity search.
- **Generator**: Gemini LLM generates answers grounded strictly in retrieved context.
- **Dynamic Prompt Builder**: Generates domain-tailored system prompts based on user description.
- **Evaluation Pipeline**: DeepEval with 5 metrics (4 built-in + 1 custom GEval).

### Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Gemini Embedding 001 |
| Vector Store | FAISS |
| Orchestration | LangChain |
| Evaluation | DeepEval |
| Frontend | Streamlit |
| Language | Python 3.10+ |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

The API key is entered directly in the Streamlit UI sidebar. No `.env` file needed.

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the chatbot

```bash
streamlit run streamlit_app.py
```

Enter your Google Gemini API key in the sidebar, upload documents, and start chatting. Use the Evaluation tab to generate QA pairs and run metrics.

## Design Decisions

1. **Gemini as unified provider**: Using Gemini for generation, embeddings, and evaluation keeps the stack simple with a single API key and zero cost on the free tier.

2. **Dynamic prompt generation**: Instead of hardcoding a domain-specific prompt, the system generates one dynamically based on the user's domain description. This makes the chatbot truly domain-agnostic.

3. **JSON output format**: The generator outputs structured JSON with answer and citations, enabling verifiable responses and a clean UI presentation.

4. **Component-level evaluation**: Retriever and generator are evaluated separately with dedicated metrics, allowing targeted debugging and optimization.

5. **Synthetic test data**: Using DeepEval's Synthesizer to generate QA pairs ensures systematic testing including edge cases, rather than relying on hand-picked questions.

## Evaluation Metrics

### Retriever Metrics
- **Contextual Recall**: Is enough information retrieved to answer the query?
- **Contextual Precision**: Are relevant chunks ranked higher than irrelevant ones?

### Generator Metrics
- **Faithfulness**: Does the answer stick to retrieved context? (anti-hallucination)
- **Answer Relevancy**: Is the answer relevant to the question asked?
- **Answer Correctness** (GEval): Is the answer factually correct based on context?

## Potential Improvements

Given additional time, the following enhancements could be made:

1. **Autonomous agent architecture**: Upgrade from a fixed RAG pipeline to a LangChain AgentExecutor with multiple tools (web search, calculator, database lookup) enabling the LLM to decide how to answer queries.

2. **Cloud evaluation dashboard**: Integrate Confident AI platform for visual side-by-side experiment comparisons, regression testing, and team collaboration on evaluation results.

3. **Production frontend**: Replace Streamlit with a React/Next.js application featuring proper chat UX, document management, and conversation history persistence.

4. **Advanced retrieval**: Add reranking models, hybrid search (keyword + semantic), and query expansion to improve retrieval quality.

5. **Multi-turn conversation memory**: Add context window management so users can ask follow-up questions that reference previous answers.

6. **Hyperparameter optimization**: Systematic A/B testing of chunk sizes, overlap, top-K, embedding dimensions, and prompt templates using the evaluation pipeline.

7. **Authentication and multi-tenancy**: User accounts with separate document collections and isolated chat histories.
