# main.py

"""
RAG Agent — Main Entry Point

Run this to interact with your RAG agent in the terminal.
Phase 1 testing. In Phase 3 this becomes a FastAPI endpoint.
In Phase 4 it becomes a Streamlit chat UI.

Usage:
    python main.py
"""

# ── Step 1: Fix Python path FIRST (before any local imports) ──────────────────
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# WHY: Python searches sys.path to find modules.
#      By inserting the project root (rag-system/) at position 0,
#      "from rag_core.graph import run_query" resolves correctly
#      regardless of which directory you run the script from.

# ── Step 2: Standard library imports ─────────────────────────────────────────
import logging

# ── Step 3: Third-party imports ───────────────────────────────────────────────
from dotenv import load_dotenv

# ── Step 4: Load env vars BEFORE local imports ────────────────────────────────
# WHY here and not inside main()?
#   Our local modules (planner, retriever etc.) read env vars at import time
#   (e.g. to initialize singleton clients). load_dotenv() must run first
#   so GROQ_API_KEY, QDRANT_URL etc. are available when those modules load.
load_dotenv()

# ── Step 5: Local imports (now resolves correctly) ────────────────────────────
from rag_core.graph import run_query  # ✅ no more red underline  # noqa: E402

# ── Step 6: Configure logging ─────────────────────────────────────────────────
# After all imports — logging config is code, not an import
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────


def print_result(result: dict) -> None:
    """Pretty print the agent's result."""
    print("\n" + "=" * 60)
    print(f"🧠 INTENT    : {result.get('intent', 'unknown').upper()}")
    print(f"📄 DOCS USED : {result.get('metadata', {}).get('docs_used', 0)}")
    print(f"\n💬 RESPONSE:\n{result.get('response', '')}")
    print("=" * 60 + "\n")


def main() -> None:
    """Interactive multi-turn chat loop."""
    print("\n🚀 RAG Agent Ready!")
    print("   Type your question and press Enter")
    print("   Type 'quit' to exit\n")

    chat_history: list = []  # Persists across turns in this session

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in {"quit", "exit", "q"}:
                print("Goodbye! 👋")
                break

            # Run the full LangGraph pipeline
            result = run_query(query, chat_history)

            # Print result to terminal
            print_result(result)

            # Carry forward conversation history for next turn
            chat_history = result.get("chat_history", [])

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logging.exception("Unexpected error in main loop")


if __name__ == "__main__":
    main()
