# AutoStream AI Sales Agent

> **Built for:** ServiceHive · Inflx Platform — Machine Learning Intern Assignment  
> **LLM:** Claude 3 Haiku · **Framework:** LangGraph (LangChain) · **Language:** Python 3.9+

A production-grade Conversational AI Agent that turns social media conversations into qualified leads for **AutoStream** — a SaaS video editing platform for content creators.

---

## Demo

```
You: Hi there!

Aria: Hey! 👋 Welcome to AutoStream — the AI-powered video editing platform
      built for content creators like you. Whether you're on YouTube, Instagram,
      or TikTok, we make professional editing effortless. What can I help you with?

You: What does the Pro plan include?

Aria: Great question! The AutoStream Pro Plan is $79/month and includes:
      - Unlimited videos per month
      - 4K resolution output
      - AI-powered captions in 50+ languages
      - 24/7 priority support
      - 500GB cloud storage + analytics dashboard
      Want me to get you started with a free trial?

You: That sounds awesome. I want to try the Pro plan for my YouTube channel.

Aria: Fantastic! Let's get you set up 🎉 Could I start with your full name?

You: Alex Johnson

Aria: Nice to meet you, Alex! What's the best email address to reach you at?

You: alex@example.com

Aria: Perfect! And just to confirm — you're primarily creating on YouTube, right?

You: Yes, YouTube.

✅  Lead captured successfully!
    Name     : Alex Johnson
    Email    : alex@example.com
    Platform : YouTube

Aria: You're all set, Alex! 🚀 Our team will reach out to alex@example.com
      shortly to activate your Pro trial. In the meantime, is there anything
      else you'd like to know?
```

---

## Project Structure

```
inflx-autostream-agent/
│
├── agent/
│   ├── __init__.py          # Package exports
│   ├── state.py             # LangGraph AgentState TypedDict
│   ├── nodes.py             # All graph node functions (intent, RAG, lead, response)
│   ├── graph.py             # Graph assembly + conditional routing
│   └── tools.py             # mock_lead_capture() tool
│
├── rag/
│   ├── __init__.py
│   └── retriever.py         # Keyword-based RAG over JSON knowledge base
│
├── knowledge_base/
│   └── autostream_kb.json   # Pricing, features, policies, FAQs
│
├── main.py                  # CLI entry point + chat loop
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Project Structure

```
inflx-autostream-agent/
├── frontend/
│   └── index.html           # Full-featured chat UI (dark theme, lead tracker)
├── agent/ ...               # LangGraph agent (see Architecture below)
├── rag/ ...                 # RAG retriever
├── knowledge_base/ ...      # JSON knowledge base
├── server.py                # FastAPI backend (serves UI + /chat API)
├── main.py                  # CLI fallback
├── start.sh                 # One-command launcher
└── README.md
```

---

## How to Run Locally

### Option A — Web UI (recommended)

```bash
chmod +x start.sh
./start.sh
# Opens http://localhost:8000
```

**Available endpoints:**
| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Chat UI |
| `http://localhost:8000/docs` | FastAPI Swagger UI |
| `http://localhost:8000/leads` | View captured leads (JSON) |
| `http://localhost:8000/health` | Health check |

### Option B — CLI only

```bash
python main.py
```

---

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/inflx-autostream-agent.git
cd inflx-autostream-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# OR
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

> Get your key at [console.anthropic.com](https://console.anthropic.com)

### 5. Run the Agent

```bash
python main.py
```

**Enable debug mode** (shows intent classification + state after each turn):

```bash
AUTOSTREAM_DEBUG=1 python main.py
```

### 6. View Captured Leads

After a successful lead capture, a `captured_leads.json` file is created in the project root:

```json
[
  {
    "lead_id": "LEAD-20240915143022",
    "name": "Alex Johnson",
    "email": "alex@example.com",
    "platform": "YouTube",
    "captured_at": "2024-09-15T14:30:22Z",
    "source": "Inflx Social Agent",
    "status": "new"
  }
]
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, inspectable state machines** — a critical requirement when building sales agents where conversation flow must be controlled and predictable. Unlike AutoGen's multi-agent conversation model, LangGraph lets us define *exactly* which node executes under which conditions using typed state and conditional edges. This prevents premature lead capture (a key assignment requirement) and makes the flow auditable.

### State Management

The agent uses a single `AgentState` TypedDict that persists **across all conversation turns** within a session. Each graph invocation receives the full state (including the complete `messages` list) and returns an updated copy. Key state fields include: `messages` (full history via `operator.add`), `intent` (classified per turn), `lead_info` (incrementally populated), `collecting_lead` (mode flag), and `lead_captured` (terminal flag).

### Graph Flow

Every user message triggers the same graph: **classify_intent → (conditional) → [rag_retrieval | collect_lead | generate_response] → generate_response**. The intent classifier uses Claude 3 Haiku to label the turn; routing edges then dispatch to the appropriate node. The `collect_lead` node extracts structured data (name, email, platform) from free-form text using a secondary LLM call, merges it into state, and only routes to `lead_capture` once all three fields are confirmed.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following integration approach would be used:

### Architecture

```
WhatsApp User
     │  (message)
     ▼
WhatsApp Business API (Meta Cloud API)
     │  POST /webhook  (JSON payload)
     ▼
FastAPI / Flask Server  (your backend)
     │  extract sender_id + message text
     ▼
Session Store (Redis)
     │  load AgentState by sender_id
     ▼
LangGraph Agent  (invoke with current state)
     │  updated state + AI reply
     ▼
Session Store (Redis)
     │  save updated AgentState
     ▼
WhatsApp Business API
     │  POST /messages  (send reply)
     ▼
WhatsApp User  (receives reply)
```

### Step-by-Step Implementation

**1. Register a WhatsApp Business App**
- Create a Meta Developer App at [developers.facebook.com](https://developers.facebook.com)
- Enable the WhatsApp Business API product
- Get your `WHATSAPP_TOKEN` and `PHONE_NUMBER_ID`

**2. Set Up a Webhook Endpoint**

```python
# webhook_server.py
from fastapi import FastAPI, Request
import httpx, redis, json
from agent import get_graph
from langchain_core.messages import HumanMessage

app = FastAPI()
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

VERIFY_TOKEN = "your_webhook_verify_token"
WA_TOKEN = "your_whatsapp_bearer_token"
PHONE_NUMBER_ID = "your_phone_number_id"

@app.get("/webhook")
async def verify(request: Request):
    """WhatsApp webhook verification handshake."""
    params = dict(request.query_params)
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(params["hub.challenge"])
    return {"error": "Forbidden"}, 403

@app.post("/webhook")
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages."""
    body = await request.json()

    try:
        entry = body["entry"][0]["changes"][0]["value"]
        msg = entry["messages"][0]
        sender_id = msg["from"]
        text = msg["text"]["body"]
    except (KeyError, IndexError):
        return {"status": "ignored"}

    # Load session state from Redis
    raw = r.get(f"session:{sender_id}")
    state = json.loads(raw) if raw else initial_state()

    # Run agent
    state["messages"] = state["messages"] + [HumanMessage(content=text)]
    graph = get_graph()
    result = graph.invoke(state)

    # Save updated state
    r.setex(f"session:{sender_id}", 3600, json.dumps(result, default=str))

    # Extract reply and send back to WhatsApp
    ai_msgs = [m for m in result["messages"] if m.type == "ai"]
    reply = ai_msgs[-1].content if ai_msgs else "Sorry, I didn't catch that."
    await send_whatsapp_message(sender_id, reply)

    return {"status": "ok"}

async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        await client.post(url, headers=headers, json=payload)
```

**3. Expose the Webhook**

For local development, use [ngrok](https://ngrok.com):
```bash
ngrok http 8000
# Copy the https:// URL → set it as webhook URL in Meta Developer Console
```

For production, deploy to AWS Lambda, Google Cloud Run, or a VPS.

**4. Session Persistence**

Each WhatsApp sender (`from` phone number) maps to a unique Redis key: `session:{sender_id}`. The full `AgentState` (serialized as JSON) is stored with a 1-hour TTL, enabling multi-turn conversations across separate webhook events — simulating the in-memory session the CLI maintains.

---

## Evaluation Checklist

| Criterion | Implementation |
|---|---|
| Intent detection | LLM-based classifier → 5 intent categories |
| RAG pipeline | Keyword-scored JSON KB retrieval → injected into system prompt |
| State management | `AgentState` TypedDict with `operator.add` messages across all turns |
| Tool calling | `mock_lead_capture()` called only after all 3 fields confirmed |
| Code clarity | Modular: nodes, graph, state, tools, RAG all separated |
| Deployability | WhatsApp webhook design documented above |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph (LangChain) |
| LLM | Claude 3 Haiku (Anthropic) |
| RAG | Custom keyword retriever over JSON KB |
| Lead Storage | Local JSON file (`captured_leads.json`) |
| Session (WhatsApp) | Redis |
| Webhook Server | FastAPI |

---

## License

MIT — built as part of the ServiceHive / Inflx ML Intern Assignment.
