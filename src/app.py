import sys
import os
import platform
import asyncio
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Fix for Windows asyncio ProactorEventLoop ConnectionResetError [WinError 10054]
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Ensure current directory is in sys.path for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from chain import Chain
from settings import KNOWLEDGE_DIR

# --- Page Configuration ---
st.set_page_config(
    page_title="Harry Potter Chat Bot | RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #740001 0%, #ae0001 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 25px rgba(116, 0, 1, 0.25);
    }
    .main-header h1 {
        color: #d4af37 !important;
        margin: 0 0 6px 0;
        font-size: 2.2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
        font-size: 1.05rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .starter-btn {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 16px;
        color: #e6edf3;
        font-size: 0.95rem;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .starter-btn:hover {
        background: #30363d;
        border-color: #d4af37;
        transform: translateY(-2px);
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialise Chain ---
@st.cache_resource
def get_chain():
    """Cache the Chain instance across app reruns for fast response time."""
    return Chain()

try:
    chain = get_chain()
    chain_loaded = True
except Exception as e:
    chain_loaded = False
    chain_error = str(e)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Harry Potter knowledge assistant powered by RAG. Ask me anything about the wizarding world! ⚡"}
    ]

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚡ Harry Chat Bot")
    st.markdown("##### ⚙️ System Control")

    if chain_loaded:
        st.markdown('<div class="status-badge">🟢 Qdrant Vector Store Connected</div>', unsafe_allow_html=True)
    else:
        st.error(f"🔴 Connection Error: {chain_error if 'chain_error' in locals() else 'Unknown'}")

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base Management")
    st.caption("Load / update documents from `src/knowledge/` into Qdrant vector database.")

    if st.button("🚀 Load Knowledge to DB", use_container_width=True):
        if chain_loaded:
            with st.spinner("Processing & embedding knowledge documents..."):
                try:
                    chain.load_knowledge_to_db()
                    st.success("Successfully ingested documents into Qdrant collection!")
                except Exception as ex:
                    st.error(f"Failed to load knowledge: {ex}")
        else:
            st.error("Chain is not properly initialised. Check environment credentials.")

    st.markdown("---")
    st.markdown("### 📁 Knowledge Files")
    if KNOWLEDGE_DIR.exists():
        files = [f.name for f in KNOWLEDGE_DIR.rglob("*") if f.is_file()]
        if files:
            for f in files:
                st.markdown(f"• `{f}`")
        else:
            st.info("No documents found in knowledge directory.")
    else:
        st.warning("Knowledge directory not found.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you? ⚡"}
        ]
        st.rerun()

# --- Main Layout ---
st.markdown("""
<div class="main-header">
    <h1>⚡ Harry Potter RAG Assistant</h1>
    <p>Ask questions powered by document context & Qdrant vector retrieval</p>
</div>
""", unsafe_allow_html=True)

# Suggested starter questions
st.markdown("##### 💡 Suggested Questions")
cols = st.columns(3)
sample_questions = [
    "Who are the four founders of Hogwarts?",
    "What are the rules and positions in Quidditch?",
    "Tell me about the properties of the Elder Wand",
]

selected_starter = None
for idx, q in enumerate(sample_questions):
    with cols[idx]:
        if st.button(q, key=f"starter_{idx}", use_container_width=True):
            selected_starter = q

# Render existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Determine user input
prompt = st.chat_input("Ask a question about the wizarding world...")
if selected_starter:
    prompt = selected_starter

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating answer..."):
            try:
                response_placeholder = st.empty()
                full_response = ""
                for token in chain.answer_question_stream(prompt):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"⚠️ An error occurred while generating answer: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
