# gateway/portkey_client.py

"""
Portkey LLM Gateway

WHY PORTKEY?
    Sits between our code and Groq.
    Provides:
    1. FALLBACK  → 70B fails → auto switch to 8B
    2. RETRIES   → rate limit hit → auto retry
    3. LOGGING   → every call logged in Portkey dashboard
    4. ONE KEY   → manage all LLM providers in one place

FALLBACK FLOW:
    Request → Llama 3.3 70B (primary)
                   │ fails (429/500/503)
                   ▼
              Llama 3.1 8B (fallback)
                   │ also fails
                   ▼
              Exception raised → handled by API

WITHOUT PORTKEY: one failure = user sees error
WITH PORTKEY:    one failure = silent fallback
"""

import os
import logging
from typing import Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_portkey_client():
    """
    Initialize Portkey client with fallback config.
    Returns None if Portkey is not configured.
    Falls back to direct Groq in that case.
    """
    try:
        from portkey_ai import Portkey

        portkey_api_key = os.getenv("PORTKEY_API_KEY")
        virtual_key_primary = os.getenv("PORTKEY_VIRTUAL_KEY_PRIMARY")
        virtual_key_fallback = os.getenv("PORTKEY_VIRTUAL_KEY_FALLBACK")

        if not portkey_api_key:
            logger.warning("[Portkey] PORTKEY_API_KEY not set — skipping gateway")
            return None

        if not virtual_key_primary:
            logger.warning("[Portkey] PORTKEY_VIRTUAL_KEY_PRIMARY not set — skipping")
            return None

        # Fallback config — try 70B first, fall back to 8B on failure
        config = {
            "strategy": {
                "mode": "fallback",
            },
            "targets": [
                {
                    # Primary — best quality
                    "virtual_key": virtual_key_primary,
                    "override_params": {
                        "model": "llama-3.3-70b-versatile",
                        "max_tokens": 2048,
                        "temperature": 0.1,
                    },
                    "on_status_codes": [429, 500, 503],
                },
                {
                    # Fallback — faster, more available
                    "virtual_key": virtual_key_fallback or virtual_key_primary,
                    "override_params": {
                        "model": "llama-3.1-8b-instant",
                        "max_tokens": 2048,
                        "temperature": 0.1,
                    },
                },
            ],
        }

        client = Portkey(
            api_key=portkey_api_key,
            config=config,
        )

        logger.info("[Portkey] Gateway initialized with fallback ✅")
        return client

    except Exception as e:
        logger.warning(f"[Portkey] Failed to initialize: {e}")
        return None


def call_llm(messages: list, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Call LLM through Portkey gateway.
    Falls back to direct Groq if Portkey is not configured.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        model:    Primary model (Portkey handles fallback automatically)

    Returns:
        Generated text response as string
    """
    client = get_portkey_client()

    # ── Path 1: Through Portkey Gateway ───────────────────────────────────────
    if client is not None:
        try:
            response: Any = client.chat.completions.create(
                messages=messages,
                model=model,
                stream=False,
            )
            content = response.choices[0].message.content
            answer = content.strip() if isinstance(content, str) else ""
            logger.info(f"[Portkey] Response via gateway ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"[Portkey] Gateway call failed: {e} — falling back to Groq")

    # ── Path 2: Direct Groq (Portkey not configured or failed) ────────────────
    logger.info("[Portkey] Using direct Groq connection")
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGroq(
            model=model,
            temperature=0.1,
            max_tokens=2048,
        )

        # Convert dict messages to LangChain message objects
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))

        response = llm.invoke(lc_messages)
        content = response.content
        return content.strip() if isinstance(content, str) else str(content[0]).strip()

    except Exception as e:
        logger.error(f"[Portkey] Direct Groq also failed: {e}")
        raise
