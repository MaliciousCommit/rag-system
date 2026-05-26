# observability/langsmith_config.py

"""
LangSmith Configuration — Agent Trace Observability

WHY LANGSMITH?
    Every time your LangGraph agent runs, LangSmith records:
    - Which nodes ran and in what order
    - What each node received as input
    - What each node returned as output
    - How long each node took
    - Token usage per LLM call
    - Full conversation context

    This is INVALUABLE for debugging. Instead of adding
    print() statements everywhere, you see everything in
    a beautiful UI at smith.langchain.com

HOW IT WORKS:
    LangChain/LangGraph automatically detects LANGCHAIN_API_KEY
    and LANGCHAIN_TRACING_V2=true in environment.
    Zero code changes needed in your agent — it just works.

    We just need to call setup_langsmith() at startup.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def setup_langsmith() -> bool:
    """
    Configure LangSmith tracing.

    Returns:
        True if LangSmith is configured, False if skipped.

    LangSmith reads these env vars automatically:
        LANGCHAIN_API_KEY      → your LangSmith API key
        LANGCHAIN_TRACING_V2   → "true" to enable tracing
        LANGCHAIN_PROJECT      → project name in dashboard
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    project = os.getenv("LANGCHAIN_PROJECT", "rag-system")

    if not api_key:
        logger.warning("[LangSmith] LANGCHAIN_API_KEY not set — tracing disabled")
        return False

    if tracing != "true":
        logger.warning("[LangSmith] LANGCHAIN_TRACING_V2 not 'true' — tracing disabled")
        return False

    # LangChain reads these automatically — just confirm they're set
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project

    logger.info(f"[LangSmith] Tracing enabled → project: '{project}' ✅")
    logger.info("[LangSmith] View traces at: https://smith.langchain.com")
    return True


def get_langsmith_url() -> str:
    """Return the LangSmith project URL for logging."""
    project = os.getenv("LANGCHAIN_PROJECT", "rag-system")
    return f"https://smith.langchain.com/projects/{project}"
