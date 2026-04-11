"""
FastAPI Backend Server for AutoStream AI Sales Agent.

Serves:
  GET  /           → frontend/index.html
  POST /chat       → runs one agent turn, returns reply + state metadata
  GET  /leads      → returns all captured leads
  POST /reset      → clears session and starts fresh

Run with:
    pip install fastapi uvicorn
    python server.py
"""

import os
import json
import uuid
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

from agent import get_graph, AgentState

app = FastAPI(title="AutoStream AI Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (keyed by session_id)
SESSIONS: dict[str, AgentState] = {}

LEADS_FILE = "captured_leads.json"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    collecting_lead: bool
    lead_captured: bool
    lead_info: dict
    missing_fields: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def initial_state() -> AgentState:
    return {
        "messages": [],
        "intent": "unknown",
        "kb_context": "",
        "lead_info": {"name": None, "email": None, "platform": None},
        "collecting_lead": False,
        "lead_captured": False,
        "missing_fields": ["name", "email", "platform"],
    }


def serialize_messages(messages):
    """Convert LangChain message objects to JSON-serializable dicts."""
    out = []
    for m in messages:
        out.append({
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content
        })
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the chat frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Frontend not found. Run the project setup first.")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process one conversation turn through the LangGraph agent."""
    session_id = req.session_id or str(uuid.uuid4())

    # Get or create session state
    if session_id not in SESSIONS:
        SESSIONS[session_id] = initial_state()

    state = SESSIONS[session_id]

    # Append user message
    state["messages"] = state["messages"] + [HumanMessage(content=req.message)]

    # Run the agent graph
    try:
        graph = get_graph()
        result = graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Save updated state
    SESSIONS[session_id] = result

    # Extract latest AI reply
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_msgs[-1].content if ai_msgs else "I'm sorry, I couldn't generate a response."

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        intent=result.get("intent", "unknown"),
        collecting_lead=result.get("collecting_lead", False),
        lead_captured=result.get("lead_captured", False),
        lead_info=result.get("lead_info") or {},
        missing_fields=result.get("missing_fields", []),
    )


@app.post("/reset")
async def reset_session(body: dict):
    """Reset a session to fresh state."""
    session_id = body.get("session_id")
    if session_id and session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"status": "reset", "session_id": session_id}


@app.get("/leads")
async def get_leads():
    """Return all captured leads from the JSON store."""
    if not os.path.exists(LEADS_FILE):
        return {"leads": [], "count": 0}
    with open(LEADS_FILE, "r") as f:
        leads = json.load(f)
    return {"leads": leads, "count": len(leads)}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("\n🚀  AutoStream Agent API starting...")
    print("    Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
