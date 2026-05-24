# rag_core/graph.py

"""
LangGraph Agent Assembly — The Complete RAG Pipeline

This file connects all nodes into a directed graph with conditional routing.

GRAPH STRUCTURE:
    START
      │
      ▼
   [Planner] ──── intent="rag" ──────────────────▶ [Retriever]
      │                                                  │
      ├──── intent="chitchat" ──────────────────────────▶│
      │                                                  │
      └──── intent="out_of_scope" ──────────────────────▶│
                                                         ▼
                                                    [Responder]
                                                         │
                                                         ▼
                                                  [MemorySaver]
                                                         │
                                                         ▼
                                                        END

WHY LangGraph over a simple function pipeline?
    1. State management — automatic state merging between nodes
    2. Conditional routing — nodes can branch based on state values
    3. Built-in streaming — stream tokens as they're generated (Phase 4)
    4. LangSmith integration — automatic trace capture (Phase 5)
    5. Checkpointing — save/resume graph execution (future feature)
"""

import logging
from typing import Optional  # ✅ FIX 2: needed for Optional[list]
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph  # ✅ FIX 1: correct return type

from rag_core.state import GraphState
from rag_core.nodes.planner import planner_node
from rag_core.nodes.retriever import retriever_node
from rag_core.nodes.responder import responder_node
from rag_core.nodes.memory_saver import memory_saver_node

logger = logging.getLogger(__name__)


def route_after_planner(state: GraphState) -> str:
    """
    Conditional routing function — called after Planner runs.

    LangGraph uses this function's RETURN VALUE to decide which node runs next.
    The return value must match one of the keys in add_conditional_edges().

    WHY only two routes (not three)?
        "chitchat" and "out_of_scope" both skip the Retriever.
        They both go directly to the Responder.
        The Responder handles them differently based on intent.
        No need to create separate nodes for each — keep it simple.
    """
    intent = state.get("intent", "rag")
    logger.info(f"[Router] Routing based on intent: '{intent}'")

    if intent == "rag":
        return "retriever"  # Needs document lookup
    else:
        return "responder"  # Chitchat and out_of_scope skip retrieval


def build_graph() -> CompiledStateGraph:  # ✅ FIX 1: was StateGraph
    """
    Assembles and compiles the complete LangGraph agent.

    Returns:
        CompiledStateGraph — ready to invoke with .invoke() or .stream()

    NOTE:
        StateGraph    = the builder  (add nodes, add edges)
        CompiledStateGraph = the runner   (invoke, stream)
        .compile() converts StateGraph → CompiledStateGraph
    """
    # ── Step 1: Create the graph with our state schema ────────────────────────
    workflow = StateGraph(GraphState)
    # WHY pass GraphState?
    #   LangGraph validates all node inputs/outputs against this schema.
    #   It also knows which fields use operator.add (chat_history).

    # ── Step 2: Register all nodes ────────────────────────────────────────────
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("responder", responder_node)
    workflow.add_node("memory_saver", memory_saver_node)
    # Each node is a Python function: (GraphState) -> dict

    # ── Step 3: Set entry point ───────────────────────────────────────────────
    workflow.set_entry_point("planner")
    # Every query starts at the Planner — no exceptions.

    # ── Step 4: Add conditional routing from Planner ──────────────────────────
    workflow.add_conditional_edges(
        "planner",  # Source node
        route_after_planner,  # Routing function
        {
            "retriever": "retriever",  # returns "retriever" → go to Retriever
            "responder": "responder",  # returns "responder" → go to Responder
        },
    )

    # ── Step 5: Add fixed edges ───────────────────────────────────────────────
    workflow.add_edge("retriever", "responder")
    # After retrieval → always go to Responder

    workflow.add_edge("responder", "memory_saver")
    # After response → always save to memory

    workflow.add_edge("memory_saver", END)
    # After saving → graph is done

    # ── Step 6: Compile ───────────────────────────────────────────────────────
    graph = workflow.compile()
    logger.info("[Graph] LangGraph compiled successfully ✅")

    return graph  # ✅ FIX 1: now matches CompiledStateGraph


def run_query(
    query: str, chat_history: Optional[list] = None
) -> dict:  # ✅ FIX 2: Optional[list]
    """
    High-level function to run a single query through the RAG agent.

    Args:
        query:        User's question
        chat_history: Previous conversation turns (for multi-turn).
                      Optional — defaults to empty list if not provided.

    Returns:
        Final GraphState dict with response, intent, retrieved docs, etc.

    Usage:
        result = run_query("What is the refund policy?")
        print(result["response"])
    """
    # ✅ FIX 1 + 3: graph is now CompiledStateGraph which HAS .invoke()
    graph = build_graph()

    initial_state = {
        "query": query,
        "intent": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": "",
        "chat_history": chat_history or [],  # None → [] safely handled here
        "metadata": {},
    }

    logger.info(f"[Graph] Running query: '{query}'")
    result = graph.invoke(
        initial_state
    )  # ✅ FIX 3: .invoke() exists on CompiledStateGraph
    logger.info("[Graph] Query complete ✅")

    return result
