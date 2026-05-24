# rag_core/nodes/memory_saver.py

"""
MemorySaver Node — Conversation History Management

WHY IS THIS A SEPARATE NODE?
    Single Responsibility Principle — each node does ONE thing.
    The Responder generates answers. The MemorySaver persists them.
    Keeping them separate means we can swap memory backends later
    (e.g., from in-memory list to Redis or a database) without
    touching the Responder at all.

IMPORTANT — How LangGraph memory works:
    We use Annotated[List[dict], operator.add] in GraphState.
    This means when this node returns {"chat_history": [new_turn]},
    LangGraph APPENDS new_turn to the existing list instead of replacing it.
    That's why we return a list with ONE item — the new turn to append.
"""

import logging
from rag_core.state import GraphState

logger = logging.getLogger(__name__)


def memory_saver_node(state: GraphState) -> dict:
    """
    MemorySaver Node — Appends the current turn to conversation history.

    Args:
        state: Full GraphState after Responder has run

    Returns:
        dict with "chat_history" containing the TWO new messages to append
        (user message + assistant response)

    NOTE on operator.add:
        If chat_history currently = [turn1, turn2]
        And we return {"chat_history": [turn3, turn4]}
        Then after this node: chat_history = [turn1, turn2, turn3, turn4]
        That's the magic of Annotated[List, operator.add]
    """
    query = state["query"]
    response = state.get("response", "")
    intent = state.get("intent", "unknown")

    logger.info(f"[MemorySaver] Saving turn | Intent: {intent}")

    # We save BOTH the user message and assistant response
    # This gives the Planner and Responder full context on next turn
    new_turns = [
        {
            "role": "user",
            "content": query,
        },
        {
            "role": "assistant",
            "content": response,
            "metadata": {"intent": intent},
        },
    ]

    logger.info(f"[MemorySaver] Appended {len(new_turns)} messages to history")

    # Return only the new turns — LangGraph appends them via operator.add
    return {"chat_history": new_turns}
