
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent
DB = BASE / "phantom.db"

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")

app = FastAPI(title="PHANTOM AI")

app.mount(
    "/static",
    StaticFiles(directory=BASE / "static"),
    name="static"
)

SYSTEM = """
You are PHANTOM AI, a professional coding assistant.
You write Python, JavaScript, HTML, CSS and other code.
You answer questions, generate ideas and analyze problems.
Explain code clearly. Never pretend to execute code.
Respond in the user's language.
"""

def connect():
    return sqlite3.connect(DB)

@app.on_event("startup")
def startup():
    with connect() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation TEXT,
            role TEXT,
            content TEXT,
            created TEXT
        )
        """)

class Chat(BaseModel):
    message: str
    conversation_id: str | None = None

@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")

@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA}/api/tags")
            r.raise_for_status()
        return {"status": "online", "model": MODEL}
    except Exception:
        return {"status": "offline", "model": MODEL}

@app.post("/api/chat")
async def chat(data: Chat):
    conversation = data.conversation_id or str(uuid.uuid4())

    with connect() as db:
        history = db.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation=? ORDER BY id DESC LIMIT 20",
            (conversation,)
        ).fetchall()

        db.execute(
            "INSERT INTO messages VALUES (NULL,?,?,?,?)",
            (conversation, "user", data.message, datetime.now().isoformat())
        )

    messages = [{"role": "system", "content": SYSTEM}]
    messages += [
        {"role": role, "content": content}
        for role, content in reversed(history)
    ]
    messages.append({"role": "user", "content": data.message})

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{OLLAMA}/api/chat",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"]
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"AI model unavailable: {error}"
        )

    with connect() as db:
        db.execute(
            "INSERT INTO messages VALUES (NULL,?,?,?,?)",
            (conversation, "assistant", answer, datetime.now().isoformat())
        )

    return {
        "conversation_id": conversation,
        "answer": answer
    }
