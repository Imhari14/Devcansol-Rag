# RAG Chatbot - Domain-Specific AI Chatbot with RAG and Evaluation

A Retrieval-Augmented Generation chatbot that answers user queries based on uploaded domain-specific documents. Built with Gemini, LangChain, FAISS, and evaluated using DeepEval.

## Architecture

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/24e84ad3-c0dd-4fc7-b90c-1706bdedf982" />


### Components

- **Document Loader**: Extracts text from TXT, PDF, and DOCX files.
- **Retriever**: Uses FAISS vector store with Gemini embeddings for semantic similarity search.
- **Generator**: Gemini LLM generates answers grounded strictly in retrieved context.
- **Dynamic Prompt Builder**: Generates domain-tailored system prompts based on user description.
- **Evaluation Pipeline**: DeepEval with 5 metrics (4 built-in + 1 custom GEval).

### Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
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

