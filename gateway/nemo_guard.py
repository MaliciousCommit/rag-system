# gateway/nemo_guard.py

import logging
from typing import Tuple  # ✅ import Tuple from typing

logger = logging.getLogger(__name__)

BLOCKED_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "ignore your instructions",
    "disregard previous",
    "forget your instructions",
    "override instructions",
    "override your instructions",
    "jailbreak",
    "dan mode",
    "developer mode",
    "god mode",
    "pretend you are",
    "pretend you're",
    "act as if you are",
    "you are now",
    "bypass",
    "do anything now",
    "repeat your instructions",
    "show me your prompt",
    "what are your instructions",
    "reveal your system prompt",
    "tell me your prompt",
    "print your instructions",
    "output your instructions",
]

BLOCKED_TOPICS = [
    "how to make a bomb",
    "how to make drugs",
    "how to synthesize",
    "how to hack",
    "illegal weapons",
    "how to kill",
]


def check_input_safety(query: str) -> Tuple[bool, str]:  # ✅ Tuple not tuple
    """
    Check if a query is safe to process.

    Returns:
        (is_safe, reason)
        True  → allow query through
        False → block query
    """
    query_lower = query.lower().strip()

    # ── Stage 1: Blocked Phrases ──────────────────────────────────────────────
    for phrase in BLOCKED_PHRASES:
        if phrase in query_lower:
            reason = f"Blocked phrase: '{phrase}'"
            logger.warning(f"[Guardrails] BLOCKED — {reason}")
            return False, reason

    # ── Stage 2: Harmful Topics ───────────────────────────────────────────────
    for topic in BLOCKED_TOPICS:
        if topic in query_lower:
            reason = f"Harmful topic: '{topic}'"
            logger.warning(f"[Guardrails] BLOCKED — {reason}")
            return False, reason

    # ── All checks passed ─────────────────────────────────────────────────────
    logger.debug("[Guardrails] Query passed ✅")
    return True, "passed"  # ✅ explicit final return — all paths covered


def get_blocked_response() -> str:
    """Standard message when a query is blocked."""
    return (
        "I'm sorry, I can't process that request. "
        "Please ask questions related to your documents."
    )
