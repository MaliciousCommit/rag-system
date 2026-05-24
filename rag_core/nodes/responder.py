# rag_core/nodes/responder.py

"""
Responder Node — Answer Generation

Takes the reranked documents + user query and generates a grounded,
accurate response using Groq's Llama 3.3 70B.

KEY PRINCIPLE — Grounded Generation:
    We ONLY answer from the provided context documents.
    If the answer isn't in the docs, we say so honestly.
    This prevents hallucination — the #1 failure mode in RAG systems.

    "Grounded" = the answer is anchored to real retrieved text,
    not to the LLM's training data (which may be outdated or wrong).
"""

import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from rag_core.state import GraphState

load_dotenv()
logger = logging.getLogger(__name__)


def get_llm() -> ChatGroq:
    """
    Initialize Groq's Llama 3.3 70B for answer generation.

    WHY 70B here (vs 8B for the Planner)?
        Answer quality matters. The 70B model:
        - Better follows complex instructions (stay grounded, cite sources)
        - Better synthesizes information from multiple document chunks
        - Better handles nuanced questions

        Temperature=0.1 (not 0):
        - Pure 0 can sound robotic
        - 0.1 adds slight natural variation without hallucination risk
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )


# ─── Prompts ──────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a precise and helpful document assistant.

Your job is to answer questions based STRICTLY on the provided context documents.

RULES:
1. Only use information from the provided context to answer
2. If the answer is not in the context, say: "I couldn't find this information in the available documents."
3. Be concise and direct — no fluff
4. If multiple documents are relevant, synthesize them coherently
5. Never make up information or use your training data for facts

FORMAT:
- Answer directly without saying "Based on the context..."
- Use bullet points for multi-part answers
- Keep responses under 300 words unless detail is essential"""

CHITCHAT_SYSTEM_PROMPT = """You are a friendly and helpful document assistant.
The user is making casual conversation. Respond warmly and briefly.
Remind them you're here to help with questions about their documents."""

OUT_OF_SCOPE_SYSTEM_PROMPT = """You are a focused document assistant.
The user's question is outside your area of expertise.
Politely decline and redirect them to ask about the documents you have access to.
Be friendly, not robotic."""


def format_context(reranked_docs: list) -> str:
    """
    Format retrieved documents into a clean context string for the LLM.

    WHY structured formatting?
        The LLM needs to clearly distinguish between different source documents.
        Numbered sections with source metadata help the LLM:
        1. Know where each piece of information comes from
        2. Avoid blending facts from unrelated documents
        3. Reference sources in its answer if needed
    """
    if not reranked_docs:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(reranked_docs, 1):
        source = doc.get("metadata", {}).get("source", f"Document {i}")
        text = doc.get("text", "")
        score = doc.get("rerank_score", 0)

        context_parts.append(
            f"[Document {i}] Source: {source} | Relevance: {score:.2f}\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def responder_node(state: GraphState) -> dict:
    """
    Responder Node — Generates the final answer based on intent + context.

    THREE PATHS:
        1. intent="rag"          → Use retrieved docs to answer
        2. intent="chitchat"     → Casual response, no docs needed
        3. intent="out_of_scope" → Polite decline

    Args:
        state: GraphState with query, intent, reranked_docs, chat_history

    Returns:
        dict with "response" key
    """
    query = state["query"]
    intent = state.get("intent", "rag")
    reranked_docs = state.get("reranked_docs", [])
    chat_history = state.get("chat_history", [])

    logger.info(f"[Responder] Generating response for intent: '{intent}'")

    llm = get_llm()

    # Build conversation history for multi-turn context
    # This lets the LLM understand follow-up questions like "tell me more"
    history_messages = []
    for turn in chat_history[-6:]:  # Last 3 turns
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        # Note: We skip assistant messages in history to save tokens
        # In Phase 3, we'll add full conversation memory

    # ── PATH 1: RAG Response ──────────────────────────────────────────────────
    if intent == "rag":
        context = format_context(reranked_docs)

        user_message = f"""Context Documents:
{context}

User Question: {query}

Answer based on the context above:"""

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            *history_messages,
            HumanMessage(content=user_message),
        ]

    # ── PATH 2: Chitchat Response ─────────────────────────────────────────────
    elif intent == "chitchat":
        messages = [
            SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
            *history_messages,
            HumanMessage(content=query),
        ]

    # ── PATH 3: Out of Scope ──────────────────────────────────────────────────
    else:
        messages = [
            SystemMessage(content=OUT_OF_SCOPE_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

    try:
        response = llm.invoke(messages)
        content = response.content
        answer = (
            content.strip() if isinstance(content, str) else str(content[0]).strip()
        )

        logger.info(f"[Responder] Generated response ({len(answer)} chars)")

        return {
            "response": answer,
            "metadata": {
                "responder_model": "llama-3.3-70b-versatile",
                "response_length": len(answer),
                "docs_used": len(reranked_docs),
            },
        }

    except Exception as e:
        logger.error(f"[Responder] LLM call failed: {e}")
        return {
            "response": "I encountered an error generating a response. Please try again.",
            "metadata": {"responder_error": str(e)},
        }
