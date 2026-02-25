"""
Streamlit frontend for the RAG Chatbot.
Provides document upload, domain configuration, chat interface, and evaluation dashboard.
"""

import os
import json
import logging
import tempfile

import streamlit as st
import pandas as pd

from app.config import validate_config, set_api_key, get_api_key
from app.document_loader import load_document, SUPPORTED_EXTENSIONS
from app.prompt_builder import generate_dynamic_prompt
from app.rag_agent import RAGAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Cansol RAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    /* --- Global --- */
    .block-container { padding-top: 0.5rem; }
    header[data-testid="stHeader"] { background: transparent; }
    .stMainBlockContainer { padding-top: 2rem; }

    /* --- Headers --- */
    .main-header {
        font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem;
        letter-spacing: -0.02em; text-align: center; padding-right: 4rem;
    }
    .sub-header {
        font-size: 0.9rem; margin-bottom: 1.5rem; line-height: 1.5;
        text-align: center; padding-right: 4rem; opacity: 0.65;
    }
    .sidebar-section-title {
        font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.4rem; opacity: 0.6;
    }

    /* --- Status Cards --- */
    .stat-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; }
    .stat-card {
        flex: 1; border: 1px solid rgba(128,128,128,0.2); border-radius: 10px;
        padding: 0.85rem 1rem; transition: box-shadow 0.2s ease;
        background: rgba(128,128,128,0.06);
    }
    .stat-card:hover { box-shadow: 0 2px 8px rgba(128,128,128,0.1); }
    .stat-label {
        font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 0.2rem; opacity: 0.5;
    }
    .stat-value { font-size: 1.35rem; font-weight: 700; }
    .stat-value-sm { font-size: 0.85rem; font-weight: 600; line-height: 1.4; opacity: 0.8; }

    /* --- Citation Box --- */
    .citation-box {
        background: rgba(128,128,128,0.06); border-left: 3px solid #6366f1;
        padding: 0.6rem 0.85rem; margin: 0.4rem 0; font-size: 0.82rem;
        border-radius: 0 6px 6px 0; line-height: 1.5; opacity: 0.85;
    }

    /* --- Token and Eval captions --- */
    div[data-testid="stChatMessage"] .stCaption p {
        font-size: 0.72rem; opacity: 0.5;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    }

    /* --- Eval metric cards --- */
    .eval-card {
        background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px; padding: 0.85rem 1rem; text-align: center;
    }
    .eval-metric-name {
        font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.3rem; opacity: 0.5;
    }
    .eval-score { font-size: 1.3rem; font-weight: 700; }
    .eval-pass { color: #22c55e; }
    .eval-fail { color: #ef4444; }
    .eval-badge {
        display: inline-block; font-size: 0.6rem; font-weight: 700;
        padding: 0.15rem 0.45rem; border-radius: 4px; text-transform: uppercase;
        letter-spacing: 0.05em; margin-left: 0.3rem; vertical-align: middle;
    }
    .badge-pass { background: rgba(34,197,94,0.15); color: #22c55e; }
    .badge-fail { background: rgba(239,68,68,0.15); color: #ef4444; }

    /* --- Chat styling --- */
    div[data-testid="stChatMessage"] { border-radius: 10px; padding: 0.75rem 1rem; }
    .stChatInput > div { border-radius: 10px; }

    /* --- Eval detail section --- */
    .eval-detail-label {
        font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.2rem; margin-top: 0.6rem; opacity: 0.5;
    }
    .eval-reason-box {
        background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.2);
        border-radius: 6px; padding: 0.6rem 0.85rem; font-size: 0.8rem;
        line-height: 1.55; margin-bottom: 0.5rem; opacity: 0.85;
    }
    .eval-metric-header {
        display: flex; align-items: center; gap: 0.5rem;
        margin-top: 0.75rem; margin-bottom: 0.25rem;
    }
    .eval-metric-title { font-size: 0.85rem; font-weight: 600; }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] .main-header { text-align: left; }
    section[data-testid="stSidebar"] .sub-header { text-align: left; }
    section[data-testid="stSidebar"] .stDivider { margin-top: 0.75rem; margin-bottom: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Session State Initialization ---
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "domain_description" not in st.session_state:
    st.session_state.domain_description = ""
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "uploaded_file_paths" not in st.session_state:
    st.session_state.uploaded_file_paths = []
if "cached_goldens" not in st.session_state:
    st.session_state.cached_goldens = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


def reset_agent():
    """Reset the agent and clear all session state."""
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.documents_loaded = False
    st.session_state.domain_description = ""
    st.session_state.chunk_count = 0
    st.session_state.doc_count = 0
    st.session_state.eval_results = None
    st.session_state.uploaded_file_paths = []
    st.session_state.cached_goldens = []
    logger.info("[frontend] agent reset by user")


# --- Sidebar: Configuration ---
with st.sidebar:
    st.markdown('<p class="main-header">Cansol RAG</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header" style="margin-bottom:0.75rem;">'
        'Configure your pipeline, upload documents, and start chatting.</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<p class="sidebar-section-title">API Key</p>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter your Google Gemini API key",
        label_visibility="collapsed",
    )

    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        set_api_key(api_key_input)
        logger.info("[frontend] API key updated from UI")
    elif st.session_state.api_key:
        set_api_key(st.session_state.api_key)

    if st.session_state.api_key:
        st.success("API key configured")
    else:
        st.warning("Enter your Google API key to get started")
        st.stop()

    st.divider()

    st.markdown('<p class="sidebar-section-title">Domain Description</p>', unsafe_allow_html=True)
    st.caption("Describe what your documents are about to tailor the chatbot.")
    domain_input = st.text_area(
        "Domain",
        value=st.session_state.domain_description,
        placeholder="e.g., Immigration law and visa application procedures.",
        label_visibility="collapsed",
        height=80,
    )

    st.divider()

    st.markdown('<p class="sidebar-section-title">Documents</p>', unsafe_allow_html=True)
    st.caption("TXT, PDF, DOCX")
    uploaded_files = st.file_uploader(
        "Documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<p class="sidebar-section-title">Retrieval Settings</p>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk size", min_value=200, max_value=2000, value=500, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=200, value=50, step=10)
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=3, step=1)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process", use_container_width=True, type="primary")
    with col2:
        reset_btn = st.button("Reset", use_container_width=True, on_click=reset_agent)


# --- Document Processing ---
if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one document.")
    else:
        with st.spinner("Processing documents..."):
            system_prompt = generate_dynamic_prompt(domain_input)
            st.session_state.domain_description = domain_input

            agent = RAGAgent(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                system_prompt=system_prompt,
            )

            texts = []
            saved_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1],
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                text = load_document(tmp_path)
                if text:
                    texts.append(text)
                    saved_paths.append(tmp_path)
                    logger.info(
                        "[frontend] document loaded - name=%s, characters=%d",
                        uploaded_file.name,
                        len(text),
                    )
                else:
                    os.unlink(tmp_path)

            if texts:
                chunk_count = agent.load_documents(texts)
                st.session_state.agent = agent
                st.session_state.documents_loaded = True
                st.session_state.chunk_count = chunk_count
                st.session_state.doc_count = len(texts)
                st.session_state.messages = []
                st.session_state.uploaded_file_paths = saved_paths
                st.sidebar.success(
                    f"Loaded {len(texts)} document(s) into {chunk_count} chunks."
                )
            else:
                st.sidebar.error("No text could be extracted from the uploaded files.")


# --- Main Content: Tabs ---
st.markdown('<p class="main-header">Cansol RAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about your uploaded documents. '
    "Answers are grounded strictly in the provided content.</p>",
    unsafe_allow_html=True,
)

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])


# =====================
# TAB 1: CHAT
# =====================
with tab_chat:
    if st.session_state.documents_loaded:
        domain_display = st.session_state.domain_description[:60] or "General"
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat-card">'
            f'<div class="stat-label">Documents</div>'
            f'<div class="stat-value">{st.session_state.doc_count}</div>'
            f'</div>'
            f'<div class="stat-card">'
            f'<div class="stat-label">Chunks</div>'
            f'<div class="stat-value">{st.session_state.chunk_count}</div>'
            f'</div>'
            f'<div class="stat-card">'
            f'<div class="stat-label">Domain</div>'
            f'<div class="stat-value-sm">{domain_display}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload documents and click 'Process Documents' in the sidebar to get started.")

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                citations = message.get("citations", [])
                if citations:
                    with st.expander("View Citations"):
                        for i, citation in enumerate(citations, 1):
                            st.markdown(
                                f'<div class="citation-box">{i}. {citation}</div>',
                                unsafe_allow_html=True,
                            )
                token_usage = message.get("token_usage")
                if token_usage and token_usage.get("total_input_tokens", 0) > 0:
                    st.caption(
                        f"Tokens -- Query: {token_usage['query_tokens']} | "
                        f"System Prompt: {token_usage['system_prompt_tokens']} | "
                        f"Retrieval Context: {token_usage['retrieval_context_tokens']} | "
                        f"Total Input: {token_usage['total_input_tokens']} | "
                        f"Output: {token_usage['output_tokens']} | "
                        f"Total: {token_usage['total_tokens']}"
                    )
                eval_scores = message.get("eval_scores")
                if eval_scores:
                    score_parts = []
                    for name, data in eval_scores.items():
                        score_parts.append(f"{name}: {data['score']:.2f}")
                    st.caption("Eval -- " + " | ".join(score_parts))

                    eval_ctx = message.get("eval_context", {})
                    with st.expander("Evaluation Details"):
                        if eval_ctx:
                            st.markdown(
                                '<p class="eval-detail-label">Input Query</p>',
                                unsafe_allow_html=True,
                            )
                            st.code(eval_ctx.get("input", ""), language=None)

                            st.markdown(
                                '<p class="eval-detail-label">Actual Output</p>',
                                unsafe_allow_html=True,
                            )
                            st.code(eval_ctx.get("actual_output", ""), language=None)

                            st.markdown(
                                '<p class="eval-detail-label">Retrieval Context</p>',
                                unsafe_allow_html=True,
                            )
                            for idx, chunk in enumerate(eval_ctx.get("retrieval_context", []), 1):
                                st.caption(f"Chunk {idx}")
                                st.code(chunk, language=None)

                            st.divider()

                        for name, data in eval_scores.items():
                            score = data["score"]
                            badge_cls = "badge-pass" if score >= 0.3 else "badge-fail"
                            badge_txt = "PASS" if score >= 0.3 else "FAIL"
                            score_cls = "eval-pass" if score >= 0.3 else "eval-fail"
                            st.markdown(
                                f'<div class="eval-metric-header">'
                                f'<span class="eval-metric-title">{name}</span>'
                                f'<span class="{score_cls}" style="font-weight:700;">{score:.2f}</span>'
                                f'<span class="eval-badge {badge_cls}">{badge_txt}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if data.get("reason"):
                                st.markdown(
                                    f'<div class="eval-reason-box">{data["reason"]}</div>',
                                    unsafe_allow_html=True,
                                )

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your documents...",
        disabled=not st.session_state.documents_loaded,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                agent = st.session_state.agent
                result = agent.answer(prompt)

                answer_text = result.get("answer", "Unable to generate an answer.")
                citations = result.get("citations", [])
                token_usage = result.get("token_usage", {})
                retrieved_context = result.get("retrieved_context", [])

                st.markdown(answer_text)

                if citations:
                    with st.expander("View Citations"):
                        for i, citation in enumerate(citations, 1):
                            st.markdown(
                                f'<div class="citation-box">{i}. {citation}</div>',
                                unsafe_allow_html=True,
                            )

                if token_usage and token_usage.get("total_input_tokens", 0) > 0:
                    st.caption(
                        f"Tokens -- Query: {token_usage['query_tokens']} | "
                        f"System Prompt: {token_usage['system_prompt_tokens']} | "
                        f"Retrieval Context: {token_usage['retrieval_context_tokens']} | "
                        f"Total Input: {token_usage['total_input_tokens']} | "
                        f"Output: {token_usage['output_tokens']} | "
                        f"Total: {token_usage['total_tokens']}"
                    )

            # Run inline evaluation metrics on the response
            eval_scores = {}
            with st.status("Evaluating response quality...", expanded=False) as eval_status:
                try:
                    import time as _time
                    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
                    from deepeval.metrics import (
                        FaithfulnessMetric,
                        AnswerRelevancyMetric,
                        GEval,
                    )
                    from deepeval.models import GeminiModel
                    from app.config import get_api_key

                    _eval_start = _time.time()

                    gemini_judge = GeminiModel(
                        model="gemini-2.5-flash",
                        api_key=get_api_key(),
                        temperature=0,
                    )

                    test_case = LLMTestCase(
                        input=prompt,
                        actual_output=answer_text,
                        retrieval_context=retrieved_context,
                    )

                    # Note: ContextualPrecision and ContextualRecall require expected_output,
                    # which is not available for live user queries. Only metrics that work
                    # without expected_output are included here.
                    inline_metrics = [
                        ("Faithfulness", FaithfulnessMetric(threshold=0.3, model=gemini_judge)),
                        ("Answer Relevancy", AnswerRelevancyMetric(threshold=0.3, model=gemini_judge)),
                        ("Answer Correctness", GEval(
                            name="Answer Correctness",
                            model=gemini_judge,
                            criteria=(
                                "Evaluate if the actual output's answer is correct and complete "
                                "based on the input and retrieved context."
                            ),
                            evaluation_params=[
                                LLMTestCaseParams.INPUT,
                                LLMTestCaseParams.ACTUAL_OUTPUT,
                                LLMTestCaseParams.RETRIEVAL_CONTEXT,
                            ],
                            threshold=0.3,
                        )),
                    ]

                    for metric_name, metric in inline_metrics:
                        st.write(f"Running {metric_name}...")
                        metric.measure(test_case)
                        score = metric.score if metric.score is not None else 0
                        eval_scores[metric_name] = {
                            "score": score,
                            "reason": metric.reason or "",
                        }

                    _elapsed = _time.time() - _eval_start
                    eval_status.update(label=f"Evaluation complete ({_elapsed:.0f}s)", state="complete")
                    logger.info("[frontend] inline eval completed - duration=%.0fs", _elapsed)

                except Exception as e:
                    eval_status.update(label="Evaluation failed", state="error")
                    logger.warning("[frontend] inline eval failed - error=%s", str(e))

            if eval_scores:
                score_parts = []
                for name, data in eval_scores.items():
                    score_parts.append(f"{name}: {data['score']:.2f}")
                st.caption("Eval -- " + " | ".join(score_parts))

                with st.expander("Evaluation Details"):
                    st.markdown(
                        '<p class="eval-detail-label">Input Query</p>',
                        unsafe_allow_html=True,
                    )
                    st.code(prompt, language=None)

                    st.markdown(
                        '<p class="eval-detail-label">Actual Output</p>',
                        unsafe_allow_html=True,
                    )
                    st.code(answer_text, language=None)

                    st.markdown(
                        '<p class="eval-detail-label">Retrieval Context</p>',
                        unsafe_allow_html=True,
                    )
                    for idx, chunk in enumerate(retrieved_context, 1):
                        st.caption(f"Chunk {idx}")
                        st.code(chunk, language=None)

                    st.divider()

                    for name, data in eval_scores.items():
                        score = data["score"]
                        badge_cls = "badge-pass" if score >= 0.3 else "badge-fail"
                        badge_txt = "PASS" if score >= 0.3 else "FAIL"
                        score_cls = "eval-pass" if score >= 0.3 else "eval-fail"
                        st.markdown(
                            f'<div class="eval-metric-header">'
                            f'<span class="eval-metric-title">{name}</span>'
                            f'<span class="{score_cls}" style="font-weight:700;">{score:.2f}</span>'
                            f'<span class="eval-badge {badge_cls}">{badge_txt}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if data.get("reason"):
                            st.markdown(
                                f'<div class="eval-reason-box">{data["reason"]}</div>',
                                unsafe_allow_html=True,
                            )

            # Store eval context alongside scores for history replay
            eval_context = {
                "input": prompt,
                "actual_output": answer_text,
                "retrieval_context": retrieved_context,
            } if eval_scores else {}

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "citations": citations,
                "token_usage": token_usage,
                "eval_scores": eval_scores,
                "eval_context": eval_context,
            })

            logger.info(
                "[frontend] response generated - query_length=%d, answer_length=%d, citations=%d",
                len(prompt),
                len(answer_text),
                len(citations),
            )


# =====================
# TAB 2: EVALUATION
# =====================
with tab_eval:
    st.markdown("**Evaluate RAG Pipeline**")
    st.caption(
        "Generate synthetic QA pairs once, then run evaluations multiple times "
        "with different settings without regenerating."
    )

    if not st.session_state.documents_loaded:
        st.info("Load documents first to run evaluation.")
    else:
        # --- Step 1: Generate or Load Goldens ---
        st.markdown("**Step 1: Synthetic QA Pairs**")

        goldens_cached = "cached_goldens" in st.session_state and st.session_state.cached_goldens
        goldens_count = len(st.session_state.cached_goldens) if goldens_cached else 0

        if goldens_cached:
            st.success(f"{goldens_count} QA pairs ready for evaluation.")

        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            generate_btn = st.button(
                "Generate QA Pairs" if not goldens_cached else "Regenerate QA Pairs",
                use_container_width=True,
            )
        with gen_col2:
            if goldens_cached:
                clear_btn = st.button("Clear QA Pairs", use_container_width=True)
                if clear_btn:
                    st.session_state.cached_goldens = []
                    st.session_state.eval_results = None
                    st.rerun()

        if generate_btn:
            import time
            from deepeval.models import GeminiModel
            from deepeval.synthesizer import Synthesizer
            from deepeval.synthesizer.config import ContextConstructionConfig
            from app.config import get_api_key
            from app.gemini_embedder import GeminiEmbeddingModel

            gemini_judge = GeminiModel(
                model="gemini-2.5-flash",
                api_key=get_api_key(),
                temperature=0,
            )

            max_retries = 3
            with st.status("Generating synthetic QA pairs...", expanded=True) as status:
                start_time = time.time()
                st.write("Initializing synthesizer...")

                for attempt in range(1, max_retries + 1):
                    try:
                        if attempt > 1:
                            wait_seconds = 10 * attempt
                            st.write(f"Retry {attempt}/{max_retries} - waiting {wait_seconds}s for API cooldown...")
                            time.sleep(wait_seconds)

                        elapsed = time.time() - start_time
                        st.write(f"Loading documents and building context... ({elapsed:.0f}s elapsed)")

                        synthesizer = Synthesizer(model=gemini_judge)

                        elapsed = time.time() - start_time
                        st.write(f"Generating QA pairs from document chunks... ({elapsed:.0f}s elapsed, est. 1-3 min)")

                        goldens = synthesizer.generate_goldens_from_docs(
                            document_paths=st.session_state.uploaded_file_paths,
                            context_construction_config=ContextConstructionConfig(
                                embedder=GeminiEmbeddingModel(),
                                critic_model=gemini_judge,
                            ),
                        )

                        elapsed = time.time() - start_time
                        st.session_state.cached_goldens = goldens
                        st.write(f"Generated {len(goldens)} QA pairs in {elapsed:.0f}s")
                        status.update(label=f"Generated {len(goldens)} QA pairs in {elapsed:.0f}s", state="complete")
                        logger.info(
                            "[frontend_eval] goldens generated and cached - count=%d, attempt=%d, duration=%.0fs",
                            len(goldens),
                            attempt,
                            elapsed,
                        )
                        st.rerun()

                    except Exception as e:
                        error_msg = str(e)
                        is_retryable = "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg
                        if is_retryable and attempt < max_retries:
                            elapsed = time.time() - start_time
                            st.write(f"API temporarily unavailable ({elapsed:.0f}s elapsed). Retrying...")
                            logger.warning(
                                "[frontend_eval] retryable error on attempt %d/%d - error=%s",
                                attempt,
                                max_retries,
                                error_msg,
                            )
                            continue
                        elapsed = time.time() - start_time
                        logger.error("[frontend_eval] golden generation failed - error=%s", error_msg)
                        if is_retryable:
                            status.update(label=f"Failed after {elapsed:.0f}s - API overloaded", state="error")
                            st.error(
                                f"Gemini API is temporarily overloaded (503). "
                                f"Tried {max_retries} times. Please wait a minute and try again."
                            )
                        else:
                            status.update(label=f"Failed after {elapsed:.0f}s", state="error")
                            st.error(f"QA pair generation failed: {error_msg}")
                        break

        # Show generated goldens preview
        if goldens_cached:
            with st.expander("Preview QA Pairs"):
                for i, golden in enumerate(st.session_state.cached_goldens):
                    st.markdown(f"**Q{i+1}:** {golden.input}")
                    if golden.expected_output:
                        st.markdown(f"**Expected:** {golden.expected_output[:200]}...")
                    st.divider()

        st.divider()

        # --- Step 2: Run Evaluation ---
        st.markdown("**Step 2: Run Evaluation**")

        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            eval_type = st.selectbox(
                "Evaluation scope",
                options=["all", "retriever", "generator"],
                index=0,
                help="Choose which components to evaluate.",
            )
        with eval_col2:
            eval_threshold = st.slider(
                "Pass threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum score for a metric to pass.",
            )

        run_eval_btn = st.button(
            "Run Evaluation",
            type="primary",
            use_container_width=True,
            disabled=not goldens_cached,
        )

        if not goldens_cached:
            st.caption("Generate QA pairs first before running evaluation.")

        if run_eval_btn and goldens_cached:
            import time
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
            from deepeval.metrics import (
                ContextualRecallMetric,
                ContextualPrecisionMetric,
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                GEval,
            )
            from deepeval.models import GeminiModel
            from deepeval import evaluate
            from app.config import get_api_key

            agent = st.session_state.agent
            goldens = st.session_state.cached_goldens

            gemini_judge = GeminiModel(
                model="gemini-2.5-flash",
                api_key=get_api_key(),
                temperature=0,
            )

            logger.info("[frontend_eval] starting evaluation with cached goldens")

            with st.status("Running evaluation pipeline...", expanded=True) as status:
                eval_start = time.time()

                try:
                    st.write(f"Creating test cases from {len(goldens)} QA pairs...")
                    test_cases = []
                    for i, golden in enumerate(goldens):
                        elapsed = time.time() - eval_start
                        st.write(f"Processing test case {i + 1}/{len(goldens)}... ({elapsed:.0f}s elapsed)")

                        retrieved_docs = agent.retrieve(golden.input)
                        result = agent.answer(golden.input)
                        actual_output = json.dumps(result) if isinstance(result, dict) else str(result)

                        test_case = LLMTestCase(
                            input=golden.input,
                            actual_output=actual_output,
                            retrieval_context=retrieved_docs,
                            expected_output=golden.expected_output,
                        )
                        test_cases.append(test_case)

                    elapsed = time.time() - eval_start
                    st.write(f"All {len(test_cases)} test cases created ({elapsed:.0f}s elapsed)")

                    metrics = []
                    if eval_type in ("retriever", "all"):
                        metrics.extend([
                            ContextualRecallMetric(threshold=eval_threshold, model=gemini_judge),
                            ContextualPrecisionMetric(threshold=eval_threshold, model=gemini_judge),
                        ])
                    if eval_type in ("generator", "all"):
                        metrics.extend([
                            FaithfulnessMetric(threshold=eval_threshold, model=gemini_judge),
                            AnswerRelevancyMetric(threshold=eval_threshold, model=gemini_judge),
                            GEval(
                                name="Answer Correctness",
                                model=gemini_judge,
                                criteria=(
                                    "Evaluate if the actual output's answer is correct and complete "
                                    "based on the input and retrieved context."
                                ),
                                evaluation_params=[
                                    LLMTestCaseParams.INPUT,
                                    LLMTestCaseParams.ACTUAL_OUTPUT,
                                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                                ],
                                threshold=eval_threshold,
                            ),
                        ])

                    st.write(
                        f"Running {len(metrics)} metrics on {len(test_cases)} test cases... "
                        f"(est. {len(test_cases) * 15}-{len(test_cases) * 25}s)"
                    )

                    eval_results = evaluate(test_cases, metrics)

                    elapsed = time.time() - eval_start
                    st.write(f"Evaluation complete in {elapsed:.0f}s")
                    status.update(
                        label=f"Evaluation complete -- {len(metrics)} metrics, {len(test_cases)} test cases, {elapsed:.0f}s",
                        state="complete",
                    )
                    logger.info("[frontend_eval] evaluation completed - duration=%.0fs", elapsed)

                    st.divider()
                    st.markdown("**Evaluation Results**")

                    metric_scores = {}
                    for tc_result in eval_results.test_results:
                        for metric_result in tc_result.metrics_data:
                            name = metric_result.name
                            if name not in metric_scores:
                                metric_scores[name] = []
                            metric_scores[name].append(metric_result.score)

                    if metric_scores:
                        cols = st.columns(min(len(metric_scores), 5))
                        for idx, (name, scores) in enumerate(metric_scores.items()):
                            valid_scores = [s for s in scores if s is not None]
                            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                            col_idx = idx % min(len(metric_scores), 5)
                            with cols[col_idx]:
                                score_cls = "eval-pass" if avg_score >= eval_threshold else "eval-fail"
                                badge_cls = "badge-pass" if avg_score >= eval_threshold else "badge-fail"
                                badge_txt = "PASS" if avg_score >= eval_threshold else "FAIL"
                                st.markdown(
                                    f'<div class="eval-card">'
                                    f'<div class="eval-metric-name">{name}</div>'
                                    f'<span class="eval-score {score_cls}">{avg_score:.2f}</span> '
                                    f'<span class="eval-badge {badge_cls}">{badge_txt}</span>'
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    st.markdown("**Detailed Results Per Test Case**")
                    table_data = []
                    for i, tc_result in enumerate(eval_results.test_results):
                        row = {"Test Case": i + 1, "Input": tc_result.input[:80] + "..."}
                        for metric_result in tc_result.metrics_data:
                            score = metric_result.score
                            row[metric_result.name] = f"{score:.2f}" if score is not None else "N/A"
                        table_data.append(row)

                    if table_data:
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    st.markdown("**Test Case Details**")
                    for i, tc_result in enumerate(eval_results.test_results):
                        with st.expander(f"Test Case {i + 1}: {tc_result.input[:60]}..."):
                            st.markdown(
                                '<p class="eval-detail-label">Input</p>',
                                unsafe_allow_html=True,
                            )
                            st.code(tc_result.input, language=None)
                            st.markdown(
                                '<p class="eval-detail-label">Output</p>',
                                unsafe_allow_html=True,
                            )
                            st.code(tc_result.actual_output[:500], language=None)
                            for metric_result in tc_result.metrics_data:
                                score = metric_result.score
                                score_str = f"{score:.2f}" if score is not None else "N/A"
                                reason = metric_result.reason or "No reason provided"
                                score_cls = "eval-pass" if score is not None and score >= eval_threshold else "eval-fail"
                                badge_cls = "badge-pass" if score is not None and score >= eval_threshold else "badge-fail"
                                badge_txt = "PASS" if score is not None and score >= eval_threshold else "FAIL"
                                st.markdown(
                                    f'<div class="eval-metric-header">'
                                    f'<span class="eval-metric-title">{metric_result.name}</span>'
                                    f'<span class="{score_cls}" style="font-weight:700;">{score_str}</span>'
                                    f'<span class="eval-badge {badge_cls}">{badge_txt}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f'<div class="eval-reason-box">{reason}</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.eval_results = metric_scores

                except Exception as e:
                    elapsed = time.time() - eval_start
                    status.update(label=f"Evaluation failed after {elapsed:.0f}s", state="error")
                    logger.error("[frontend_eval] evaluation failed - error=%s", str(e))
                    st.error(f"Evaluation failed: {str(e)}")

        # Show previous results if available
        elif st.session_state.eval_results is not None and not run_eval_btn:
            st.markdown("**Previous Evaluation Results**")
            scores = st.session_state.eval_results
            cols = st.columns(min(len(scores), 5))
            for idx, (name, score_list) in enumerate(scores.items()):
                valid = [s for s in score_list if s is not None]
                avg = sum(valid) / len(valid) if valid else 0
                col_idx = idx % min(len(scores), 5)
                with cols[col_idx]:
                    score_cls = "eval-pass" if avg >= 0.3 else "eval-fail"
                    st.markdown(
                        f'<div class="eval-card">'
                        f'<div class="eval-metric-name">{name}</div>'
                        f'<span class="eval-score {score_cls}">{avg:.2f}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
