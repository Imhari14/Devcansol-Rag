"""
Dynamic prompt generation module.
Builds tailored system prompts based on user-provided domain descriptions.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GENERATION_MODEL, get_api_key

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the query using ONLY the provided context. "
    "Do not use any outside knowledge. If the context does not contain relevant "
    "information, respond with: No relevant information available.\n\n"
    "Format your response strictly as a JSON object with the following structure:\n"
    '{\n'
    '  "answer": "<a concise, complete answer to the query>",\n'
    '  "citations": [\n'
    '    "<exact verbatim quote from the context that supports your answer>"\n'
    '  ]\n'
    '}\n\n'
    "Citation rules:\n"
    "- Each citation MUST be a direct word-for-word excerpt from the context, not a section title or heading.\n"
    "- Include enough surrounding text to make the citation meaningful (at least one full sentence).\n"
    "- Do NOT cite section numbers or headings alone.\n"
    "- Combine related information into a single citation rather than splitting into many small ones.\n"
    "- Use a maximum of 3 citations per response.\n\n"
    "Only respond in JSON. No explanations outside the JSON structure."
)

META_PROMPT_TEMPLATE = (
    "The user has uploaded documents about the following domain: \"{domain_description}\"\n\n"
    "Generate a concise system prompt for a QA assistant that:\n"
    "1. Is an expert in this specific domain\n"
    "2. Only answers from provided context, never from outside knowledge\n"
    "3. Uses appropriate tone and terminology for this domain\n"
    "4. Responds strictly in JSON format with 'answer' and 'citations' fields\n"
    "5. Says 'No relevant information available.' when context lacks the answer\n"
    "6. Citations must be exact verbatim quotes from the context (full sentences, not section titles). Combine related info into one citation. Maximum 3 citations per response.\n\n"
    "Return ONLY the system prompt text. No explanations or metadata."
)


def generate_dynamic_prompt(domain_description: str) -> str:
    """Generate a domain-specific system prompt using the LLM.

    Takes a user-provided domain description and uses Gemini to create
    a tailored system prompt with appropriate tone and terminology.

    Args:
        domain_description: A brief description of the document domain.

    Returns:
        A system prompt string tailored to the domain.
    """
    if not domain_description or not domain_description.strip():
        logger.info("[prompt_builder] no domain description provided, using default prompt")
        return DEFAULT_SYSTEM_PROMPT

    logger.info(
        "[prompt_builder] generating dynamic prompt - domain=%s",
        domain_description[:80],
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL,
            google_api_key=get_api_key(),
            temperature=0.3,
        )
        meta_prompt = META_PROMPT_TEMPLATE.format(domain_description=domain_description)
        response = llm.invoke(meta_prompt)
        generated_prompt = response.content.strip()

        logger.info(
            "[prompt_builder] dynamic prompt generated - length=%d",
            len(generated_prompt),
        )
        return generated_prompt

    except Exception as e:
        logger.error(
            "[prompt_builder] failed to generate dynamic prompt - error=%s",
            str(e),
        )
        logger.info("[prompt_builder] falling back to default prompt")
        return DEFAULT_SYSTEM_PROMPT
