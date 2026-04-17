"""
LangGraph State Definition for the AutoStream Sales Agent.
"""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class LeadInfo(TypedDict):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class AgentState(TypedDict):
    """
    The full state of the agent, persisted across all conversation turns.

    Fields:
        messages        : Full conversation history (HumanMessage + AIMessage).
        intent          : Classified intent of the latest user message.
                          One of: "greeting", "inquiry", "high_intent", "lead_collection", "unknown"
        kb_context      : Retrieved RAG context for the current turn.
        lead_info       : Collected lead fields (name, email, platform).
        collecting_lead : Whether the agent is currently in lead-collection mode.
        lead_captured   : Whether mock_lead_capture has been called successfully.
        missing_fields  : Which lead fields are still needed.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    intent: str
    kb_context: str
    lead_info: LeadInfo
    collecting_lead: bool
    lead_captured: bool
    missing_fields: List[str]
