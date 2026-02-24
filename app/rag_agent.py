"""
RAG Agent module.
Combines document retrieval and answer generation into a unified pipeline.
"""

import json
import logging
from typing import Any, Optional

from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    get_api_key,
)
from app.prompt_builder import DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RAGAgent:
    """Retrieval-Augmented Generation agent for domain-specific QA.

    Attributes:
        chunk_size: Number of characters per text chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        top_k: Number of documents to retrieve per query.
        system_prompt: The system prompt used for answer generation.
        embedding_model: The embedding model instance.
        vector_store: The FAISS vector store instance.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the RAG agent.

        Args:
            chunk_size: Size of text chunks for splitting documents.
            chunk_overlap: Overlap between consecutive chunks.
            top_k: Number of top results to retrieve.
            system_prompt: Custom system prompt for generation.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.vector_store = None

        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=get_api_key(),
            task_type="RETRIEVAL_DOCUMENT",
        )

        self.llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL,
            google_api_key=get_api_key(),
            temperature=0,
        )

        self._genai_client = genai.Client(api_key=get_api_key())

        logger.info(
            "[rag_agent] initialized - chunk_size=%d, chunk_overlap=%d, top_k=%d",
            self.chunk_size,
            self.chunk_overlap,
            self.top_k,
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's native tokenizer.

        Args:
            text: The text to count tokens for.

        Returns:
            Token count from Gemini API.
        """
        try:
            result = self._genai_client.models.count_tokens(
                model=GENERATION_MODEL,
                contents=text,
            )
            return result.total_tokens
        except Exception as e:
            logger.warning("[rag_agent] token counting failed - error=%s", str(e))
            return 0

    def load_documents(self, texts: list[str]) -> int:
        """Process and store documents in the vector store.

        Args:
            texts: List of raw text strings from documents.

        Returns:
            Total number of chunks created and stored.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        all_chunks = []
        for i, text in enumerate(texts):
            chunks = splitter.create_documents([text])
            all_chunks.extend(chunks)
            logger.info(
                "[rag_agent] document chunked - doc_index=%d, chunks_created=%d",
                i,
                len(chunks),
            )

        self.vector_store = FAISS.from_documents(all_chunks, self.embedding_model)

        total_chunks = len(all_chunks)
        logger.info(
            "[rag_agent] vector store built - total_chunks=%d",
            total_chunks,
        )
        return total_chunks

    def retrieve(self, query: str) -> list[str]:
        """Retrieve relevant document chunks for a given query.

        Args:
            query: The user's question or search query.

        Returns:
            List of relevant text chunks from the vector store.
        """
        if self.vector_store is None:
            logger.warning("[rag_agent] retrieve called with no documents loaded")
            return []

        docs = self.vector_store.similarity_search(query, k=self.top_k)
        context = [doc.page_content for doc in docs]

        logger.info(
            "[rag_agent] retrieval completed - query_length=%d, results=%d",
            len(query),
            len(context),
        )
        return context

    def generate(self, query: str, retrieved_docs: list[str]) -> str:
        """Generate an answer using the LLM based on retrieved context.

        Args:
            query: The user's question.
            retrieved_docs: List of relevant text chunks.

        Returns:
            The LLM-generated response string.
        """
        context = "\n---\n".join(retrieved_docs)

        prompt = (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Query:\n{query}"
        )

        response = self.llm.invoke(prompt)
        raw_output = response.content.strip()

        logger.info(
            "[rag_agent] generation completed - query_length=%d, context_chunks=%d, output_length=%d",
            len(query),
            len(retrieved_docs),
            len(raw_output),
        )
        return raw_output

    def answer(self, query: str) -> dict[str, Any]:
        """Run the full RAG pipeline: retrieve then generate.

        Args:
            query: The user's question.

        Returns:
            Dictionary with 'answer', 'citations', 'retrieved_context',
            and 'token_usage' keys.
        """
        retrieved_docs = self.retrieve(query)

        if not retrieved_docs:
            return {
                "answer": "No documents have been loaded. Please upload documents first.",
                "citations": [],
                "retrieved_context": [],
                "token_usage": {},
            }

        # Build the full prompt for token counting
        context_text = "\n---\n".join(retrieved_docs)
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Query:\n{query}"
        )

        # Count input tokens using Gemini's native tokenizer
        query_tokens = self._count_tokens(query)
        context_tokens = self._count_tokens(context_text)
        system_prompt_tokens = self._count_tokens(self.system_prompt)
        total_input_tokens = self._count_tokens(full_prompt)

        # Generate the answer
        raw_output = self.generate(query, retrieved_docs)

        # Count output tokens
        output_tokens = self._count_tokens(raw_output)

        token_usage = {
            "query_tokens": query_tokens,
            "system_prompt_tokens": system_prompt_tokens,
            "retrieval_context_tokens": context_tokens,
            "total_input_tokens": total_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_input_tokens + output_tokens,
            "retrieval_chunks": len(retrieved_docs),
        }

        logger.info(
            "[rag_agent] token usage - query=%d, context=%d, total_input=%d, output=%d, total=%d",
            token_usage["query_tokens"],
            token_usage["retrieval_context_tokens"],
            token_usage["total_input_tokens"],
            token_usage["output_tokens"],
            token_usage["total_tokens"],
        )

        try:
            cleaned = raw_output
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            parsed["retrieved_context"] = retrieved_docs
            parsed["token_usage"] = token_usage
            return parsed

        except json.JSONDecodeError:
            logger.warning(
                "[rag_agent] failed to parse JSON response, returning raw output"
            )
            return {
                "answer": raw_output,
                "citations": [],
                "retrieved_context": retrieved_docs,
                "token_usage": token_usage,
            }
