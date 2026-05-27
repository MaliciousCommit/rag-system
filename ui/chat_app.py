# ui/chat_app.py

import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
    .stApp { background-color: #0e1117; }
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
    .intent-rag {
        background-color: #1a472a; color: #4caf50;
        padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: bold;
    }
    .intent-chitchat {
        background-color: #1a3a4a; color: #29b6f6;
        padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: bold;
    }
    .intent-blocked {
        background-color: #4a1a1a; color: #ef5350;
        padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: bold;
    }
    .intent-out_of_scope {
        background-color: #3a3a1a; color: #ffb74d;
        padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: bold;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State ────────────────────────────────────────────────────────────
def init_session():
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
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def send_query(query: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "top_k": 8,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Try a simpler query."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is FastAPI running?"}
    except Exception as e:
        return {"error": str(e)}


def get_intent_badge(intent: str) -> str:
    return f'<span class="intent-{intent}">{intent.upper()}</span>'


def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} Source(s) Used", expanded=False):
        for source in sources:
            st.markdown(
                f"""
            <div style="background:#1a1a2e; border:1px solid #2d2d4e;
                        border-radius:8px; padding:10px; margin:4px 0; font-size:13px;">
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

    # API Status
    st.subheader("🔌 API Status")
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.info(f"API URL: {API_URL}")

    st.markdown("---")

    # Session Info
    st.subheader("📊 Session Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Files", len(st.session_state.ingested_files))
    st.caption(f"Session: `{st.session_state.session_id}`")

    st.markdown("---")

    # Document Upload
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "pptx", "html"],
        help="Upload document to knowledge base",
    )

    if uploaded_file is not None:
        st.success(f"✅ Ready: {uploaded_file.name}")

        if st.button("⚡ Ingest Document", use_container_width=True):
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                try:
                    # ✅ Send bytes to FastAPI — no local imports needed
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    }
                    response = requests.post(
                        f"{API_URL}/ingest",
                        files=files,
                        timeout=30,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result.get('message', 'Ingestion started!')}")
                        if uploaded_file.name not in st.session_state.ingested_files:
                            st.session_state.ingested_files.append(uploaded_file.name)
                    else:
                        st.error(f"❌ {response.text}")

                except Exception as e:
                    st.error(f"❌ {str(e)}")

    if st.session_state.ingested_files:
        st.markdown("**Ingested Files:**")
        for f in st.session_state.ingested_files:
            st.caption(f"📄 {f}")

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
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


# ─── Main Chat Area ───────────────────────────────────────────────────────────
st.title("💬 Chat with Your Documents")

if not check_api_health():
    st.warning(f"⚠️ FastAPI is not running.\n\nAPI URL: `{API_URL}`")
    st.stop()

# Chat History
if not st.session_state.messages:
    st.markdown(
        """
    <div style="text-align:center; padding:50px; color:#666;">
        <h3>👋 Welcome!</h3>
        <p>Upload a document in the sidebar, then ask me anything!</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-message">👤 {message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
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
        if message.get("sources"):
            render_sources(message["sources"])

    st.markdown("<br>", unsafe_allow_html=True)

# Input
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

# Handle Send
if send_clicked and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🤔 Thinking..."):
        result = send_query(query)

    if "error" in result:
        st.error(f"❌ {result['error']}")
        st.session_state.messages.pop()
    else:
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
