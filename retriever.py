"""
RAG Retriever for AutoStream Knowledge Base.
Uses keyword-based semantic search over a structured JSON knowledge base.
"""

import json
import os
import re
from typing import List, Tuple


KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "autostream_kb.json")


def load_knowledge_base() -> dict:
    """Load the knowledge base from disk."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_kb(kb: dict) -> List[Tuple[str, str]]:
    """
    Flatten the knowledge base into (topic, text) pairs for retrieval.
    Returns a list of (category_label, full_text) tuples.
    """
    chunks = []

    # Company info
    company = kb.get("company", {})
    chunks.append((
        "company_overview",
        f"Company: {company.get('name')}\n"
        f"Description: {company.get('description')}\n"
        f"Tagline: {company.get('tagline')}"
    ))

    # Plans
    for plan in kb.get("plans", []):
        features_str = "\n  - ".join(plan["features"])
        chunks.append((
            f"plan_{plan['name'].lower().replace(' ', '_')}",
            f"Plan Name: {plan['name']}\n"
            f"Price: {plan['price']}\n"
            f"Best For: {plan['best_for']}\n"
            f"Features:\n  - {features_str}"
        ))

    # Policies
    for policy in kb.get("policies", []):
        chunks.append((
            f"policy_{policy['topic'].lower().replace(' ', '_')}",
            f"Policy – {policy['topic']}:\n{policy['details']}"
        ))

    # FAQs
    for faq in kb.get("faqs", []):
        chunks.append((
            "faq",
            f"Q: {faq['question']}\nA: {faq['answer']}"
        ))

    return chunks


def _score_chunk(query: str, chunk_text: str) -> float:
    """
    Simple keyword overlap scoring between query and chunk.
    Returns a relevance score (higher = more relevant).
    """
    query_tokens = set(re.findall(r'\w+', query.lower()))
    chunk_tokens = set(re.findall(r'\w+', chunk_text.lower()))

    # Stopwords to ignore
    stopwords = {"i", "a", "the", "is", "are", "what", "how", "can", "do",
                 "me", "my", "you", "your", "it", "in", "on", "for", "to",
                 "about", "and", "or", "of", "with", "have", "has", "be",
                 "tell", "give", "show", "get", "want", "need"}

    query_tokens -= stopwords
    if not query_tokens:
        return 0.0

    overlap = query_tokens & chunk_tokens
    score = len(overlap) / len(query_tokens)

    # Boost score for exact phrase matches
    for token in query_tokens:
        if token in chunk_text.lower():
            score += 0.05

    return score


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Retrieve the most relevant knowledge base chunks for a given query.

    Args:
        query: The user's question or message.
        top_k: Number of top chunks to return.

    Returns:
        A formatted string of the most relevant knowledge base sections.
    """
    kb = load_knowledge_base()
    chunks = _flatten_kb(kb)

    # Score all chunks
    scored = [(label, text, _score_chunk(query, text)) for label, text in chunks]
    scored.sort(key=lambda x: x[2], reverse=True)

    # Filter chunks with non-zero score; fall back to top chunks if all zero
    relevant = [(label, text, score) for label, text, score in scored if score > 0]
    if not relevant:
        relevant = scored  # fallback: return top_k anyway

    top_chunks = relevant[:top_k]

    if not top_chunks:
        return "No relevant information found in the knowledge base."

    result_parts = []
    for label, text, score in top_chunks:
        result_parts.append(f"[Source: {label}]\n{text}")

    return "\n\n---\n\n".join(result_parts)


def get_full_kb_summary() -> str:
    """Return a concise summary of all knowledge base content (for system prompt context)."""
    kb = load_knowledge_base()
    chunks = _flatten_kb(kb)
    return "\n\n".join(text for _, text in chunks)
