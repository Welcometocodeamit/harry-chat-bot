import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Ensure current directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from chain import Chain
from settings import KNOWLEDGE_DIR

# ---------------------------------------------------------------------------
# Application lifecycle — initialise the Chain once on startup
# ---------------------------------------------------------------------------
chain_instance: Chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain_instance
    print("==================================================")
    print("Starting application: Initialising Chain ...")
    print("==================================================")
    try:
        chain_instance = Chain()
        print("[SUCCESS] Chain initialised and ready!")
    except Exception as e:
        print(f"[ERROR] Failed to initialise Chain: {e}")
    yield
    print("Shutting down application ...")


app = FastAPI(
    title="Harry Chat Bot",
    description="RAG-powered Q&A API with Qdrant vector search over the Harry Potter universe",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Embedded HTML Chat UI
# ---------------------------------------------------------------------------
HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>Hogwarts Library Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #0a0a0c;
            --bg-secondary: #141418;
            --bg-card: #1c1c22;
            --bg-chat-user: #1f2937;
            --bg-chat-bot: #291215;
            --accent: #d4af37;
            --accent-hover: #b8860b;
            --accent-glow: rgba(212, 175, 55, 0.35);
            --accent-secondary: #740001;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --border-color: #2e3a4e;
            --success-color: #10b981;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        html, body {
            height: 100%; height: 100dvh; width: 100vw; overflow: hidden;
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary); color: var(--text-main);
        }

        body { display: flex; position: fixed; inset: 0; }

        /* Sidebar */
        .sidebar {
            width: 320px; background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex; flex-direction: column; padding: 24px; gap: 20px; flex-shrink: 0;
        }
        .brand-logo { display: flex; align-items: center; gap: 12px; }
        .brand-title {
            font-family: 'Cinzel', serif; font-size: 1.3rem; font-weight: 700;
            color: var(--accent); letter-spacing: 0.5px;
        }
        .status-pill {
            display: flex; align-items: center; gap: 8px;
            background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25);
            padding: 8px 14px; border-radius: 20px; font-size: 0.82rem;
            color: var(--success-color); font-weight: 500;
        }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background-color: var(--success-color); box-shadow: 0 0 10px var(--success-color);
        }
        .sidebar-section {
            background-color: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 16px;
        }
        .sidebar-section h3 {
            font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--text-muted); margin-bottom: 12px;
        }
        .file-list {
            list-style: none; display: flex; flex-direction: column; gap: 8px;
            max-height: 180px; overflow-y: auto;
        }
        .file-item {
            font-size: 0.85rem; color: #cbd5e1; display: flex; align-items: center; gap: 8px;
            background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 6px; word-break: break-all;
        }
        .btn {
            background: linear-gradient(135deg, var(--accent), #6366f1); color: white; border: none;
            padding: 12px 16px; border-radius: 10px; font-weight: 600; font-size: 0.9rem;
            cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center;
            justify-content: center; gap: 8px; width: 100%;
        }
        .btn:hover {
            background: linear-gradient(135deg, var(--accent-hover), #4f46e5);
            box-shadow: 0 4px 15px var(--accent-glow); transform: translateY(-1px);
        }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }

        /* Mobile overlay */
        .sidebar-overlay {
            display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7);
            backdrop-filter: blur(2px); z-index: 25; opacity: 0; transition: opacity 0.3s ease;
        }
        .sidebar-overlay.open { display: block; opacity: 1; }

        /* Main chat area */
        .main-container {
            flex: 1; display: flex; flex-direction: column; height: 100%; height: 100dvh;
            width: 100%; min-width: 0; min-height: 0; background-color: var(--bg-primary); position: relative;
        }
        .chat-header {
            padding: 20px 32px; background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color); display: flex; align-items: center; gap: 16px;
        }
        .menu-btn {
            display: none; background: none; border: none; color: var(--accent);
            cursor: pointer; padding: 4px; border-radius: 6px;
        }
        .menu-btn svg { width: 28px; height: 28px; fill: currentColor; }
        .chat-header-text { display: flex; flex-direction: column; }
        .chat-header h2 { font-family: 'Cinzel', serif; font-size: 1.5rem; font-weight: 600; color: var(--accent); }

        .chat-messages {
            flex: 1; overflow-y: auto; -webkit-overflow-scrolling: touch;
            padding: 20px 32px; display: flex; flex-direction: column; gap: 18px; min-height: 0;
        }
        .message { display: flex; gap: 16px; max-width: 85%; animation: fadeIn 0.3s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .avatar {
            width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center;
            justify-content: center; font-weight: 700; font-size: 0.9rem; flex-shrink: 0;
        }
        .message.assistant .avatar {
            background: linear-gradient(135deg, #740001, #ae0001); color: #d4af37;
            box-shadow: 0 0 12px rgba(116,0,1,0.5); border: 1px solid #d4af37;
        }
        .message.user .avatar { background: #3b82f6; color: white; }
        .message-content {
            background-color: var(--bg-card); border: 1px solid var(--border-color);
            padding: 14px 18px; border-radius: 16px; line-height: 1.6; font-size: 0.95rem;
        }
        .message.user .message-content {
            background-color: #2563eb; border-color: #3b82f6; color: white; border-bottom-right-radius: 4px;
        }
        .message.assistant .message-content { border-bottom-left-radius: 4px; }
        .message-content p { margin-bottom: 8px; }
        .message-content p:last-child { margin-bottom: 0; }

        /* Starter chips */
        .starters-wrapper {
            display: flex; gap: 10px; padding: 0 32px 12px; overflow-x: auto; flex-shrink: 0;
            scrollbar-width: none;
        }
        .starters-wrapper::-webkit-scrollbar { display: none; }
        .starter-chip {
            background: var(--bg-card); border: 1px solid var(--border-color);
            padding: 8px 14px; border-radius: 20px; font-size: 0.85rem;
            color: var(--text-muted); cursor: pointer; white-space: nowrap; transition: all 0.2s ease;
        }
        .starter-chip:hover { border-color: var(--accent); color: white; background: rgba(124,58,237,0.1); }

        /* Input area */
        .chat-input-area {
            padding: 12px 32px max(16px, env(safe-area-inset-bottom));
            background-color: var(--bg-secondary); border-top: 1px solid var(--border-color);
            flex-shrink: 0; width: 100%;
        }
        .input-box-wrapper {
            display: flex; gap: 12px; background-color: var(--bg-card);
            border: 1px solid var(--border-color); border-radius: 14px; padding: 6px 6px 6px 16px;
            transition: border-color 0.2s ease;
        }
        .input-box-wrapper:focus-within { border-color: var(--accent); box-shadow: 0 0 12px var(--accent-glow); }
        .chat-input {
            flex: 1; background: none; border: none; color: white; font-size: 0.95rem;
            outline: none; font-family: inherit;
        }
        .send-btn {
            width: 44px; height: 44px; background: linear-gradient(135deg, var(--accent), #6366f1);
            border: none; border-radius: 10px; color: white; cursor: pointer;
            display: flex; align-items: center; justify-content: center; transition: all 0.2s ease;
        }
        .send-btn:hover {
            background: linear-gradient(135deg, var(--accent-hover), #4f46e5);
            box-shadow: 0 4px 12px var(--accent-glow);
        }
        .send-btn svg { width: 20px; height: 20px; fill: currentColor; }

        /* Typing indicator */
        .typing-indicator { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
        .typing-indicator span {
            width: 6px; height: 6px; background-color: var(--text-muted);
            border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both;
        }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                position: fixed; top: 0; left: 0; bottom: 0; width: min(300px, 80vw);
                transform: translateX(-100%); z-index: 30;
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 5px 0 25px rgba(0,0,0,0.7);
            }
            .sidebar.open { transform: translateX(0); }
            .menu-btn { display: block; }
            .message { max-width: 92%; }
            .chat-messages { padding: 14px; gap: 14px; }
            .chat-header { padding: 14px; }
            .chat-input-area { padding: 10px 14px max(12px, env(safe-area-inset-bottom)); }
            .starters-wrapper { padding: 0 14px 10px; }
            .chat-input { font-size: 16px; }
        }
    </style>
</head>
<body>
    <div class="sidebar-overlay" id="sidebarOverlay" onclick="toggleSidebar()"></div>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="brand-logo">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <path d="M22 4L12 18H16L10 28L24 12H18L22 4Z" fill="#d4af37"/>
            </svg>
            <span class="brand-title">Hogwarts Library</span>
        </div>
        <div class="status-pill">
            <div class="status-dot"></div>
            <span>Vector Store Connected</span>
        </div>
        <div class="sidebar-section">
            <h3>Knowledge Files</h3>
            <ul class="file-list" id="fileList">
                <li class="file-item">Loading directory content...</li>
            </ul>
        </div>
    </div>

    <!-- Main Chat View -->
    <div class="main-container">
        <div class="chat-header">
            <button class="menu-btn" onclick="toggleSidebar()">
                <svg viewBox="0 0 24 24"><path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/></svg>
            </button>
            <div class="chat-header-text">
                <h2>The Restricted Section</h2>
                <span style="font-size: 0.85rem; color: var(--text-muted);">Magical Knowledge Retrieval System</span>
            </div>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message assistant">
                <div class="avatar">⚡</div>
                <div class="message-content">
                    Greetings! I am your magical assistant, ready to delve into the Restricted Section of the library. What do you wish to know about the wizarding world?
                </div>
            </div>
        </div>

        <!-- Starter chips -->
        <div class="starters-wrapper">
            <div class="starter-chip" onclick="sendStarter('Who are the four founders of Hogwarts?')">Who founded Hogwarts?</div>
            <div class="starter-chip" onclick="sendStarter('What are the rules and positions in Quidditch?')">Rules of Quidditch</div>
            <div class="starter-chip" onclick="sendStarter('Tell me about the properties of the Elder Wand')">The Elder Wand</div>
        </div>

        <!-- Input -->
        <div class="chat-input-area">
            <div class="input-box-wrapper">
                <input type="text" id="userInput" class="chat-input" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                    <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        function toggleSidebar() {
            document.querySelector('.sidebar').classList.toggle('open');
            document.getElementById('sidebarOverlay').classList.toggle('open');
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const fileListEl = document.getElementById('fileList');
                fileListEl.innerHTML = '';
                if (data.files && data.files.length > 0) {
                    data.files.forEach(f => {
                        const li = document.createElement('li');
                        li.className = 'file-item';
                        li.textContent = '📄 ' + f;
                        fileListEl.appendChild(li);
                    });
                } else {
                    fileListEl.innerHTML = '<li class="file-item">No files in /knowledge</li>';
                }
            } catch (err) { console.error(err); }
        }
        fetchStatus();

        function sendStarter(text) {
            document.getElementById('userInput').value = text;
            sendMessage();
        }

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendMessage();
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const question = input.value.trim();
            if (!question) return;

            input.value = '';
            appendMessage('user', question);

            const assistantMsgDiv = appendMessage('assistant', '<div class="typing-indicator"><span></span><span></span><span></span></div>');
            const contentDiv = assistantMsgDiv.querySelector('.message-content');

            try {
                const response = await fetch('/api/answer-stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                if (!response.ok) throw new Error('Failed to get answer from API');

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let accumulatedText = '';
                contentDiv.innerHTML = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    accumulatedText += decoder.decode(value, { stream: true });
                    contentDiv.innerHTML = marked.parse(accumulatedText);
                    scrollToBottom();
                }
            } catch (err) {
                try {
                    const res = await fetch('/api/answer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    const data = await res.json();
                    contentDiv.innerHTML = marked.parse(data.answer || 'No answer received.');
                } catch (fallbackErr) {
                    contentDiv.innerHTML = '⚠️ Error: ' + fallbackErr.message;
                }
            }
            scrollToBottom();
        }

        function appendMessage(role, text) {
            const container = document.getElementById('chatMessages');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = role === 'assistant' ? '⚡' : '🧙';
            const content = document.createElement('div');
            content.className = 'message-content';
            content.innerHTML = role === 'user' ? escapeHtml(text) : text;
            msgDiv.appendChild(avatar);
            msgDiv.appendChild(content);
            container.appendChild(msgDiv);
            scrollToBottom();
            return msgDiv;
        }

        function scrollToBottom() {
            const container = document.getElementById('chatMessages');
            container.scrollTop = container.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def redirect_to_app():
    return HTMLResponse(content='<script>window.location.href="/app";</script>')


@app.get("/app", response_class=HTMLResponse)
def serve_ui():
    """Serve the Chatbot Web UI."""
    return HTMLResponse(content=HTML_UI)


@app.post("/api/answer")
def answer_question_api(req: QuestionRequest):
    """Return the full answer as JSON."""
    if chain_instance is None:
        raise HTTPException(status_code=500, detail="Chain instance is not initialised")
    answer = chain_instance.answer_question(req.question)
    return {"question": req.question, "answer": answer}


@app.post("/api/answer-stream")
def answer_question_stream_api(req: QuestionRequest):
    """Stream answer tokens for real-time UI rendering."""
    if chain_instance is None:
        raise HTTPException(status_code=500, detail="Chain instance is not initialised")

    def generate():
        for token in chain_instance.answer_question_stream(req.question):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/api/load-knowledge")
def load_knowledge_api():
    """Reload / ingest knowledge files into the Qdrant vector database."""
    if chain_instance is None:
        raise HTTPException(status_code=500, detail="Chain instance is not initialised")
    try:
        chain_instance.load_knowledge_to_db()
        return {"status": "success", "message": "Knowledge documents successfully loaded into Qdrant!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load knowledge: {e}")


@app.get("/api/status")
def status_api():
    """Check server status and list knowledge directory files."""
    files = []
    if KNOWLEDGE_DIR.exists():
        files = [f.name for f in KNOWLEDGE_DIR.rglob("*") if f.is_file()]
    return {
        "status": "online",
        "chain_initialized": chain_instance is not None,
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "files": files,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
