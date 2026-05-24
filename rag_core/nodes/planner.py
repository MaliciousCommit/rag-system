# rag_core/nodes/planner.py

"""
Planner Node — Intent Classification

WHY WE NEED THIS:
    Without a planner, EVERY query hits Qdrant.
    "Hello, how are you?" would trigger a vector search — wasteful and slow.
    "What is the capital of France?" is general knowledge — Qdrant won't help.

    The Planner saves compute by routing only REAL knowledge-base questions
    to the Retriever. Everything else goes straight to the Responder.

ROUTING LOGIC:
    "rag"          → Query about our documents → Retriever → Responder
    "chitchat"     → Casual talk              → Responder (no retrieval)
    "out_of_scope" → Unrelated question       → Responder (polite decline)
"""

import json
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from rag_core.state import GraphState

# Load environment variables from .env file
load_dotenv()

# Set up logging — so we can see what the planner decided in the terminal
logger = logging.getLogger(__name__)


def get_llm() -> ChatGroq:
    """
    Initialize the Groq LLM for intent classification.

    WHY Llama 3.1 8B for the Planner (not 70B)?
        Classification is a SIMPLE task — just label the query.
        Using the smaller 8B model here saves cost and latency.
        We save the powerful 70B for the Responder where quality matters.

        This is a classic AI engineering pattern:
        "Use the smallest model that can do the job."
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",  # Small + fast for classification
        temperature=0,  # 0 = deterministic, no creativity
        max_tokens=50,  # Classification only needs a few tokens
    )


# ─── System Prompt ────────────────────────────────────────────────────────────
# This prompt is carefully engineered. Notice:
# 1. We give EXACT output format (JSON) — prevents the LLM from rambling
# 2. We give clear definitions for each intent
# 3. We give examples — few-shot prompting dramatically improves accuracy
# 4. We say "respond ONLY with JSON" — prevents preamble like "Sure! Here is..."

PLANNER_SYSTEM_PROMPT = """You are an intent classification system for a document Q&A assistant.

Your ONLY job is to classify the user's query into exactly one of these intents:

1. "rag"          - Query about specific documents, policies, procedures, or
                    knowledge base content that requires document retrieval.
                    Examples: "What is the refund policy?", "How do I reset my password?",
                              "Summarize the Q3 report", "What does the contract say about..."

2. "chitchat"     - Casual conversation, greetings, or general questions not
                    requiring document lookup.
                    Examples: "Hello!", "How are you?", "What's your name?",
                              "Tell me a joke", "Thanks!"

3. "out_of_scope" - Questions completely unrelated to the document knowledge base
                    that the assistant cannot or should not answer.
                    Examples: "What is the weather today?", "Write me a poem",
                              "What stocks should I buy?"

Respond ONLY with a JSON object in this exact format:
{"intent": "<rag|chitchat|out_of_scope>", "reason": "<one sentence why>"}

No preamble. No explanation outside the JSON. Just the JSON object."""


def planner_node(state: GraphState) -> dict:
    """
    Planner Node — Classifies user intent and routes the graph.

    Args:
        state: Current GraphState containing the user's query

    Returns:
        dict with "intent" key — LangGraph merges this into the state

    FLOW:
        1. Read query from state
        2. Build chat history context (for multi-turn awareness)
        3. Call Groq (Llama 3.1 8B) with classification prompt
        4. Parse JSON response
        5. Return {"intent": "rag"|"chitchat"|"out_of_scope"}
    """
    query = state["query"]
    chat_history = state.get("chat_history", [])

    logger.info(f"[Planner] Classifying query: '{query}'")

    # Build context from chat history so the planner understands
    # follow-up questions correctly.
    # Example: "Tell me more about that" — without history context,
    # the planner can't know "that" refers to a document topic.
    history_context = ""
    if chat_history:
        # Only use last 3 turns to keep the prompt short
        recent = chat_history[-6:]  # 3 turns × 2 messages (user + assistant)
        history_lines = []
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_lines.append(f"{role.upper()}: {content}")
        history_context = "\n\nRecent conversation:\n" + "\n".join(history_lines)

    # Build the classification prompt
    user_message = f"Query to classify: {query}{history_context}"

    try:
        llm = get_llm()

        # Send to Groq
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        content = response.content
        raw_output = (
            content.strip() if isinstance(content, str) else str(content[0]).strip()
        )

        logger.debug(f"[Planner] Raw LLM output: {raw_output}")

        # Parse the JSON response
        # WHY try/except here?
        #   LLMs occasionally deviate from format instructions.
        #   Without error handling, a bad response crashes the entire graph.
        #   We catch it and default to "rag" — better to over-retrieve than fail.
        parsed = json.loads(raw_output)
        intent = parsed.get("intent", "rag")
        reason = parsed.get("reason", "")

        # Validate the intent is one of our expected values
        valid_intents = {"rag", "chitchat", "out_of_scope"}
        if intent not in valid_intents:
            logger.warning(
                f"[Planner] Unexpected intent '{intent}', defaulting to 'rag'"
            )
            intent = "rag"

        logger.info(f"[Planner] Intent: '{intent}' | Reason: {reason}")

        return {
            "intent": intent,
            "metadata": {
                "planner_reason": reason,
                "planner_model": "llama-3.1-8b-instant",
            },
        }

    except json.JSONDecodeError as e:
        # LLM didn't return valid JSON — fail safe
        logger.error(f"[Planner] JSON parse error: {e}. Defaulting to 'rag'")
        return {"intent": "rag", "metadata": {"planner_error": str(e)}}

    except Exception as e:
        # Any other error (network, rate limit, etc.)
        logger.error(f"[Planner] Unexpected error: {e}")
        return {"intent": "rag", "metadata": {"planner_error": str(e)}}
