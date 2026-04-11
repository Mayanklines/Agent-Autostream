"""
AutoStream Conversational AI Agent Main Entry Point.

Run with:
    python main.py

Environment variables required:
    ANTHROPIC_API_KEY  — Your Anthropic API key

Optional:
    AUTOSTREAM_DEBUG=1 — Show internal state after each turn
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import get_graph, AgentState

load_dotenv()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_env() -> None:
    """Ensure required environment variables are set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌  Error: ANTHROPIC_API_KEY is not set.")
        print("   Add it to a .env file or export it in your shell:")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Initial State
# ---------------------------------------------------------------------------

def initial_state() -> AgentState:
    """Return a fresh agent state."""
    return {
        "messages": [],
        "intent": "unknown",
        "kb_context": "",
        "lead_info": {"name": None, "email": None, "platform": None},
        "collecting_lead": False,
        "lead_captured": False,
        "missing_fields": ["name", "email", "platform"],
    }


# ---------------------------------------------------------------------------
# CLI Chat Loop
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════╗
║       AutoStream — AI Sales Agent (Aria)         ║
║       Powered by Inflx · Built with LangGraph    ║
╚══════════════════════════════════════════════════╝
Type your message below. Type 'quit' or 'exit' to end the session.
"""

DEBUG = os.getenv("AUTOSTREAM_DEBUG", "0") == "1"


def run_cli() -> None:
    """Run the interactive CLI chat loop."""
    check_env()
    print(BANNER)

    graph = get_graph()
    state = initial_state()
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("\nAria: Thanks for chatting! Feel free to reach out anytime. Goodbye! 👋")
            break

        turn += 1

        # Append user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph
        try:
            result = graph.invoke(state)
        except Exception as e:
            print(f"\n❌ Agent error: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            continue

        # Update state with result
        state = result

        # Extract and print the latest AI message
        ai_messages = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"]
        if ai_messages:
            latest_reply = ai_messages[-1].content
            print(f"\nAria: {latest_reply}\n")
        else:
            print("\nAria: (no response generated)\n")

        # Debug output
        if DEBUG:
            print(f"  [DEBUG] intent={state.get('intent')} | "
                  f"collecting={state.get('collecting_lead')} | "
                  f"captured={state.get('lead_captured')} | "
                  f"lead_info={state.get('lead_info')} | "
                  f"missing={state.get('missing_fields')}")
            print()

        # Session complete after lead capture
        if state.get("lead_captured"):
            print("─" * 52)
            print("✅  Lead successfully captured and saved.")
            print("    Check captured_leads.json for the record.")
            print("─" * 52)
            # Allow a few more turns for questions
            if turn > 10:
                break


if __name__ == "__main__":
    run_cli()
