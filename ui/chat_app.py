# ui/chat_app.py

"""
Streamlit Chat UI — RAG System Frontend

Talks directly to your FastAPI /query and /ingest endpoints.
Run with: streamlit run ui/chat_app.py

FEATURES:
    ✅ Chat interface with message history
    ✅ Document upload + ingestion
    ✅ Source documents display
    ✅ Session management
    ✅ Intent badge (RAG/chitchat/out_of_scope)
    ✅ Response metadata (latency, model)
    ✅ Works on laptop + phone browser
"""

import os
import sys
import uuid
import requests
import streamlit as st
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Config ───────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── Page Setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #1e3a5f, #2d5a8e);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }

    .bot-message {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: #e0e0e0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border-left: 3px solid #4a9eff;
        word-wrap: break-word;
    }

    /* Intent badge */
    .intent-rag {
        background-color: #1a472a;
        color: #4caf50;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }

    .intent-chitchat {
        background-color: #1a3a4a;
        color: #29b6f6;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }

    .intent-blocked {
        background-color: #4a1a1a;
        color: #ef5350;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }

    .intent-out_of_scope {
        background-color: #3a3a1a;
        color: #ffb74d;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }

    /* Source card */
    .source-card {
        background-color: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 8px;
        padding: 10px;
        margin: 4px 0;
        font-size: 13px;
    }

    /* Upload area */
    .upload-section {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Metrics */
    .metric-box {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        color: #888;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Input box */
    .stTextInput input {
        background-color: #1a1a2e !important;
        color: white !important;
        border: 1px solid #2d2d4e !important;
        border-radius: 12px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State ────────────────────────────────────────────────────────────


def init_session():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []


init_session()


# ─── Helper Functions ─────────────────────────────────────────────────────────


def check_api_health() -> bool:
    """Check if FastAPI is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def send_query(query: str) -> dict:
    """Send query to FastAPI /query endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "top_k": 8,
            },
            timeout=60,  # RAG can take up to 60s for complex queries
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Try a simpler query."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure FastAPI is running."}
    except Exception as e:
        return {"error": str(e)}


def ingest_document(file_path: str) -> dict:
    """Send document to FastAPI /ingest endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/ingest",
            json={
                "file_path": file_path,
                "upload_to_gcs": False,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_intent_badge(intent: str) -> str:
    """Return colored HTML badge for intent."""
    return f'<span class="intent-{intent}">{intent.upper()}</span>'


def render_sources(sources: list):
    """Render source documents in an expander."""
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} Source(s) Used", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(
                f"""
            <div class="source-card">
                <b>📄 {source.get("source", "Unknown")}</b>
                &nbsp;|&nbsp; Page {source.get("page", "?")}
                &nbsp;|&nbsp; Score: {source.get("score", 0):.3f}
                <br><br>
                <i>{source.get("text", "")[:200]}...</i>
            </div>
            """,
                unsafe_allow_html=True,
            )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.markdown("---")

    # ── API Status ────────────────────────────────────────────────────────────
    st.subheader("🔌 API Status")
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.info("Run: `python api/main.py`")

    st.markdown("---")

    # ── Session Info ──────────────────────────────────────────────────────────
    st.subheader("📊 Session Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Files", len(st.session_state.ingested_files))

    st.caption(f"Session ID: `{st.session_state.session_id}`")

    st.markdown("---")

    # ── Document Upload ───────────────────────────────────────────────────────
    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "pptx", "html", "txt"],
        help="Upload a document to add to your knowledge base",
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        save_dir = Path("temp_uploads")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / uploaded_file.name

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("⚡ Ingest Document", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                result = ingest_document(str(save_path))

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.success("✅ Ingestion started!")
                st.caption(result.get("message", ""))
                if uploaded_file.name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(uploaded_file.name)

    # Show ingested files
    if st.session_state.ingested_files:
        st.markdown("**Ingested Files:**")
        for f in st.session_state.ingested_files:
            st.caption(f"📄 {f}")

    st.markdown("---")

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        # Clear server-side session too
        try:
            requests.delete(
                f"{API_URL}/session/{st.session_state.session_id}",
                timeout=5,
            )
        except Exception:
            pass
        st.rerun()

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("---")
    st.caption("Built with LangGraph + Groq + Qdrant")
    st.caption("Phase 4 — Streamlit UI")


# ─── Main Chat Area ───────────────────────────────────────────────────────────

st.title("💬 Chat with Your Documents")

# API offline warning
if not api_healthy:
    st.warning(
        "⚠️ FastAPI is not running. Open a new terminal and run: `python api/main.py`"
    )
    st.stop()

# ── Chat History ──────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown(
            """
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>👋 Welcome!</h3>
            <p>Upload a document in the sidebar, then ask me anything about it.</p>
            <p>I can also answer general questions!</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        if message["role"] == "user":
            # User message — right aligned
            st.markdown(
                f'<div class="user-message">👤 {message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Assistant message — left aligned
            intent = message.get("intent", "rag")
            badge = get_intent_badge(intent)
            latency = message.get("latency", 0)

            st.markdown(
                f'<div class="bot-message">'
                f"🤖 &nbsp;{badge}&nbsp;&nbsp;"
                f'<span style="color:#555; font-size:11px">⏱️ {latency}s</span>'
                f"<br><br>{message['content']}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Show sources if available
            if message.get("sources"):
                render_sources(message["sources"])

        st.markdown("<br>", unsafe_allow_html=True)


# ── Input Area ────────────────────────────────────────────────────────────────
st.markdown("---")

col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        label="Message",
        placeholder="Ask anything about your documents...",
        label_visibility="collapsed",
        key="user_input",
    )

with col2:
    send_clicked = st.button("Send 🚀", use_container_width=True)


# ── Handle Send ───────────────────────────────────────────────────────────────

if send_clicked and user_input.strip():
    query = user_input.strip()

    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    # Call API
    with st.spinner("🤔 Thinking..."):
        result = send_query(query)

    if "error" in result:
        st.error(f"❌ {result['error']}")
        st.session_state.messages.pop()  # Remove user message on error
    else:
        # Add assistant message to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("answer", ""),
                "intent": result.get("intent", "rag"),
                "sources": result.get("sources", []),
                "latency": result.get("metadata", {}).get("latency_seconds", 0),
            }
        )
        st.session_state.total_queries += 1

    st.rerun()
