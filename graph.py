"""
LangGraph Graph Builder for the AutoStream Sales Agent.

Graph topology:
  START
    └─► classify_intent
          ├─► [greeting / unknown]  ──────────────────► generate_response ──► END
          ├─► [inquiry]             ──► rag_retrieval ──► generate_response ──► END
          ├─► [high_intent]         ──► collect_lead  ──► generate_response ──► END
          └─► [lead_collection]     ──► collect_lead
                                          ├─► [fields missing] ──► generate_response ──► END
                                          └─► [all fields]     ──► lead_capture
                                                                      └──► generate_response ──► END
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    classify_intent_node,
    rag_retrieval_node,
    collect_lead_node,
    lead_capture_node,
    generate_response_node,
)


# ---------------------------------------------------------------------------
# Routing Functions (Conditional Edges)
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    """Route to the appropriate node based on classified intent."""
    intent = state.get("intent", "unknown")

    if intent == "inquiry":
        return "rag_retrieval"
    elif intent in ("high_intent",):
        return "collect_lead"
    elif intent == "lead_collection":
        return "collect_lead"
    else:
        # greeting or unknown
        return "generate_response"


def route_after_collect(state: AgentState) -> str:
    """After collecting lead info, decide whether to capture or keep asking."""
    missing = state.get("missing_fields", ["name", "email", "platform"])
    if not missing:
        return "lead_capture"
    return "generate_response"


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph state graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("collect_lead", collect_lead_node)
    graph.add_node("lead_capture", lead_capture_node)
    graph.add_node("generate_response", generate_response_node)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Edges from classify_intent (conditional)
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "rag_retrieval": "rag_retrieval",
            "collect_lead": "collect_lead",
            "generate_response": "generate_response",
        }
    )

    # After RAG → generate response
    graph.add_edge("rag_retrieval", "generate_response")

    # After collect_lead (conditional)
    graph.add_conditional_edges(
        "collect_lead",
        route_after_collect,
        {
            "lead_capture": "lead_capture",
            "generate_response": "generate_response",
        }
    )

    # After lead capture → generate response
    graph.add_edge("lead_capture", "generate_response")

    # Terminal edge
    graph.add_edge("generate_response", END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    """Return the singleton compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
