# ⚡ Harry Chat Bot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about the Harry Potter universe. It loads PDF/text documents into a [Qdrant](https://qdrant.tech/) vector database, retrieves relevant passages at query time, and generates answers using an LLM via [OpenRouter](https://openrouter.ai/).

Built with **LangChain**, **FastAPI**, **Streamlit**, and **Qdrant**.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.142+-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)

---

## 📖 About

Harry Chat Bot implements a full RAG pipeline:

1. **Load** — PDF, TXT, MD, and CSV documents are loaded from the `src/knowledge/` directory
2. **Chunk** — Documents are split into overlapping chunks (900 chars, 120 overlap) for better retrieval
3. **Embed** — Each chunk is embedded using OpenAI-compatible embeddings (via OpenRouter)
4. **Store** — Embeddings are stored in a Qdrant vector database with deterministic deduplication IDs
5. **Retrieve** — At query time, the top-K most similar chunks are retrieved
6. **Generate** — Retrieved context + user question are sent to an LLM, which generates a grounded answer

### Two UI Options

| Interface | Description | Command |
|-----------|-------------|---------|
| **FastAPI Web UI** | Beautiful dark-themed chat interface served at `/app` with streaming responses | `python main.py` |
| **Streamlit UI** | Feature-rich UI with sidebar controls, knowledge management, and chat history | `streamlit run app.py` |

---

## 🏗️ Project Structure

```
harry-chat-bot/
├── main.py                  # Entry point → FastAPI server
├── app.py                   # Entry point → Streamlit app
├── pyproject.toml           # Dependencies & project metadata
├── .env.example             # Environment variable template
├── .python-version          # Python 3.12
├── .gitignore
│
└── src/
    ├── main.py              # FastAPI app (routes, embedded chat UI)
    ├── app.py               # Streamlit app (chat UI with sidebar)
    ├── chain.py             # RAG chain orchestrator
    ├── settings.py          # Shared config (knowledge directory path)
    │
    ├── loader/
    │   └── loader.py        # Document loader (PDF, TXT, MD, CSV)
    │
    ├── chunker/
    │   └── chunker.py       # Text splitter with deterministic chunk IDs
    │
    ├── llm/
    │   ├── llm.py           # LLM wrapper (OpenRouter)
    │   └── prompt.py        # RAG prompt template
    │
    ├── store/
    │   ├── embedding.py     # Embedding model wrapper
    │   └── store.py         # Qdrant vector store manager
    │
    └── knowledge/
        └── README.md        # Place your documents here
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`
- **Qdrant** — either local (Docker) or [Qdrant Cloud](https://cloud.qdrant.io/)
- **OpenRouter API key** — sign up at [openrouter.ai](https://openrouter.ai/)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/harry-chat-bot.git
cd harry-chat-bot
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

| Variable | Description | Example |
|----------|-------------|---------|
| `OPEN_ROUTER_API_KEY` | Your OpenRouter API key | `sk-or-v1-...` |
| `OPEN_API_BASE` | OpenRouter base URL | `https://openrouter.ai/api/v1` |
| `EMBEDDING_MODEL` | Embedding model ID | `openai/text-embedding-3-small` |
| `LLM_MODEL` | LLM model ID from OpenRouter | `deepseek/deepseek-v4.1-flash` |
| `LLM_BASE_URL` | LLM API base URL | `https://openrouter.ai/api/v1` |
| `QDRANT_CLIENT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key (cloud only) | *(leave empty for local)* |
| `SEARCH_K` | Number of chunks to retrieve | `15` |
| `SEARCH_TYPE` | Search strategy | `similarity` or `mmr` |

### 3. Install dependencies

**With uv (recommended):**
```bash
uv sync
```

**With pip:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .
```

### 4. Start Qdrant (local)

If using Qdrant locally via Docker:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or skip this step if you're using [Qdrant Cloud](https://cloud.qdrant.io/) (update `QDRANT_CLIENT_URL` and `QDRANT_API_KEY` in `.env`).

### 5. Add knowledge documents

Place your PDF, TXT, MD, or CSV files into the `src/knowledge/` directory:

```bash
# Example: copy a Harry Potter PDF
cp ~/Downloads/harrypotter.pdf src/knowledge/
```

### 6. Run the application

**Option A — FastAPI (recommended for API + web UI):**

```bash
# With uv
uv run python main.py

# With pip/venv
python main.py
```

Then open **http://localhost:8000/app** in your browser.

**Option B — Streamlit (feature-rich UI):**

```bash
# With uv
uv run streamlit run app.py

# With pip/venv
streamlit run app.py
```

### 7. Load knowledge into the vector database

- **FastAPI UI:** The knowledge loads are managed via the API. Send a POST request:
  ```bash
  curl -X POST http://localhost:8000/api/load-knowledge
  ```
- **Streamlit UI:** Click the **"🚀 Load Knowledge to DB"** button in the sidebar.

This step embeds all documents and stores them in Qdrant. You only need to do this once (or when you add new documents).

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/app` | Web chat UI |
| `POST` | `/api/answer` | Get a full answer (JSON) |
| `POST` | `/api/answer-stream` | Stream answer tokens (text/plain) |
| `POST` | `/api/load-knowledge` | Ingest documents into Qdrant |
| `GET` | `/api/status` | Health check + list knowledge files |

### Example API call

```bash
curl -X POST http://localhost:8000/api/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Who are the four founders of Hogwarts?"}'
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | [OpenRouter](https://openrouter.ai/) (DeepSeek, GPT, Claude, etc.) |
| **Embeddings** | OpenAI `text-embedding-3-small` via OpenRouter |
| **Vector DB** | [Qdrant](https://qdrant.tech/) |
| **Framework** | [LangChain](https://langchain.com/) |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) |
| **UI** | [Streamlit](https://streamlit.io/) + embedded HTML/CSS/JS |
| **PDF Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |

---

## 📝 License

This project is for educational and personal use.
