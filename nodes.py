"""
LangGraph Node Functions for the AutoStream Sales Agent.

Each function represents a node in the state graph and returns
a partial state update dict.
"""

import re
import os
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from .state import AgentState, LeadInfo
from .tools import mock_lead_capture
from rag.retriever import retrieve, get_full_kb_summary


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _get_llm() -> ChatAnthropic:
    """Return a Claude 3 Haiku instance."""
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.3,
        max_tokens=512,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Aria, the friendly and knowledgeable sales assistant for AutoStream — \
an AI-powered video editing SaaS platform for content creators.

Your personality: warm, professional, concise, and helpful.

Your primary goals:
1. Answer questions about AutoStream's plans, features, and policies accurately using the knowledge base.
2. Identify when a user is ready to sign up (high intent).
3. Collect name, email, and creator platform before capturing the lead.
4. Never make up information — only use provided knowledge base context.

AutoStream Knowledge Base:
{kb_summary}

Rules:
- Be conversational and natural — avoid robotic responses.
- When collecting lead info, ask for ONE missing field at a time.
- Only confirm lead capture after all three fields (name, email, platform) are collected.
- Do not ask for lead info unless the user shows clear purchase/sign-up intent.
"""


def _build_system_message() -> SystemMessage:
    kb_summary = get_full_kb_summary()
    return SystemMessage(content=SYSTEM_PROMPT.format(kb_summary=kb_summary))


# ---------------------------------------------------------------------------
# Node 1: Intent Classification
# ---------------------------------------------------------------------------

INTENT_CLASSIFY_PROMPT = """Classify the intent of the latest user message into exactly one of:
- "greeting"      : casual hello, how are you, etc.
- "inquiry"       : asking about features, pricing, policies, or general product info
- "high_intent"   : user clearly wants to sign up, buy, try, start, or subscribe
- "lead_collection": user is providing their name, email, or platform info
- "unknown"       : anything else

User message: "{message}"
Conversation so far (last 4 turns): {history}

Respond with ONLY the intent label, nothing else.
"""


def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """Classify the intent of the latest user message."""
    llm = _get_llm()

    latest_msg = state["messages"][-1].content if state["messages"] else ""

    # Build recent history string (last 4 messages)
    recent = state["messages"][-5:-1] if len(state["messages"]) > 1 else []
    history_str = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Aria'}: {m.content}"
        for m in recent
    )

    prompt = INTENT_CLASSIFY_PROMPT.format(
        message=latest_msg,
        history=history_str or "None yet"
    )

    # If already collecting lead, override to lead_collection if user is giving info
    if state.get("collecting_lead"):
        # Check if the message looks like data provision
        email_pattern = r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b'
        if re.search(email_pattern, latest_msg):
            return {"intent": "lead_collection"}

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    # Normalize to valid intents
    valid_intents = {"greeting", "inquiry", "high_intent", "lead_collection", "unknown"}
    intent = raw if raw in valid_intents else "unknown"

    # If already collecting lead and intent is not clear, keep lead_collection
    if state.get("collecting_lead") and intent not in ("lead_collection", "high_intent"):
        intent = "lead_collection"

    return {"intent": intent}


# ---------------------------------------------------------------------------
# Node 2: RAG Retrieval
# ---------------------------------------------------------------------------

def rag_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve relevant knowledge base content for the current query."""
    latest_msg = state["messages"][-1].content if state["messages"] else ""
    context = retrieve(latest_msg, top_k=3)
    return {"kb_context": context}


# ---------------------------------------------------------------------------
# Node 3: Extract Lead Info from Message
# ---------------------------------------------------------------------------

EXTRACT_LEAD_PROMPT = """Extract any of the following from the user message and return as JSON.
If a field is not present, set it to null.

Fields to extract:
- name: person's full name
- email: email address
- platform: content platform (YouTube, Instagram, TikTok, Twitter, Facebook, LinkedIn, etc.)

User message: "{message}"

Return ONLY valid JSON, e.g.: {{"name": "Alice", "email": null, "platform": "YouTube"}}
"""


def _extract_lead_info_from_message(message: str) -> Dict[str, Any]:
    """Use the LLM to extract lead fields from a user message."""
    llm = _get_llm()
    prompt = EXTRACT_LEAD_PROMPT.format(message=message)
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        data = json.loads(raw)
        return {
            "name": data.get("name"),
            "email": data.get("email"),
            "platform": data.get("platform"),
        }
    except Exception:
        # Fallback: regex email extraction
        email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b', message)
        return {
            "name": None,
            "email": email_match.group(0) if email_match else None,
            "platform": None,
        }


def _get_missing_fields(lead_info: LeadInfo) -> list:
    """Return list of lead fields that are still missing."""
    missing = []
    if not lead_info.get("name"):
        missing.append("name")
    if not lead_info.get("email"):
        missing.append("email")
    if not lead_info.get("platform"):
        missing.append("platform")
    return missing


def collect_lead_node(state: AgentState) -> Dict[str, Any]:
    """Extract lead info from the latest message and update lead_info."""
    latest_msg = state["messages"][-1].content if state["messages"] else ""
    extracted = _extract_lead_info_from_message(latest_msg)

    # Merge with existing lead_info (don't overwrite already-captured fields)
    current = dict(state.get("lead_info") or {"name": None, "email": None, "platform": None})
    for field in ("name", "email", "platform"):
        if extracted.get(field) and not current.get(field):
            current[field] = extracted[field]

    missing = _get_missing_fields(current)
    return {
        "lead_info": current,
        "collecting_lead": True,
        "missing_fields": missing,
    }


# ---------------------------------------------------------------------------
# Node 4: Lead Capture Tool Execution
# ---------------------------------------------------------------------------

def lead_capture_node(state: AgentState) -> Dict[str, Any]:
    """Call the mock lead capture API once all fields are collected."""
    lead = state["lead_info"]
    result = mock_lead_capture(
        name=lead["name"],
        email=lead["email"],
        platform=lead["platform"],
    )
    return {
        "lead_captured": True,
        "collecting_lead": False,
    }


# ---------------------------------------------------------------------------
# Node 5: Response Generation
# ---------------------------------------------------------------------------

RESPONSE_CONTEXT_TEMPLATE = """[INTERNAL CONTEXT – do not reveal this to the user]
Intent: {intent}
Collecting Lead: {collecting_lead}
Lead Captured: {lead_captured}
Lead Info So Far: {lead_info}
Missing Fields: {missing_fields}
KB Context: {kb_context}
[END INTERNAL CONTEXT]

Instructions for this turn:
{instructions}
"""


def _build_instructions(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    collecting = state.get("collecting_lead", False)
    captured = state.get("lead_captured", False)
    missing = state.get("missing_fields", ["name", "email", "platform"])

    if captured:
        lead = state.get("lead_info", {})
        return (
            f"Lead capture is complete for {lead.get('name')}. "
            f"Thank them warmly, confirm their details, and let them know the AutoStream team will be in touch. "
            f"Offer any final questions they might have."
        )

    if collecting and missing:
        next_field = missing[0]
        field_prompts = {
            "name": "Politely ask for the user's full name.",
            "email": "Ask for their email address so the team can reach them.",
            "platform": "Ask which content platform they primarily create on (e.g., YouTube, Instagram, TikTok)."
        }
        return field_prompts.get(next_field, "Ask for the next required piece of information.")

    if intent == "high_intent":
        return (
            "The user wants to sign up! Express enthusiasm, briefly confirm the Pro plan benefits, "
            "and start the lead collection by asking for their full name."
        )

    if intent == "inquiry":
        return (
            "Answer the user's product/pricing question using ONLY the KB context provided. "
            "Be accurate and concise. End with a soft call-to-action if appropriate."
        )

    if intent == "greeting":
        return (
            "Greet the user warmly as Aria from AutoStream. "
            "Briefly introduce what AutoStream does and invite them to ask questions."
        )

    return "Respond helpfully to the user's message."


def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """Generate the agent's reply using the LLM with full context."""
    llm = _get_llm()
    system = _build_system_message()

    instructions = _build_instructions(state)

    # Build internal context injection
    context_msg = RESPONSE_CONTEXT_TEMPLATE.format(
        intent=state.get("intent", "unknown"),
        collecting_lead=state.get("collecting_lead", False),
        lead_captured=state.get("lead_captured", False),
        lead_info=json.dumps(state.get("lead_info") or {}),
        missing_fields=state.get("missing_fields", []),
        kb_context=state.get("kb_context", "N/A"),
        instructions=instructions,
    )

    # Build message list: system + history + internal context hint
    messages = [system] + state["messages"] + [HumanMessage(content=context_msg)]

    response = llm.invoke(messages)
    ai_message = AIMessage(content=response.content)

    return {"messages": [ai_message]}
