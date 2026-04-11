#!/usr/bin/env bash
# start.sh — Launch the AutoStream Agent with Web UI
set -e

if [ ! -f ".env" ]; then
  echo "⚠️  No .env file found. Copying from .env.example..."
  cp .env.example .env
  echo "   → Edit .env and add your ANTHROPIC_API_KEY, then re-run."
  exit 1
fi

echo ""
echo "🎬  AutoStream AI Sales Agent"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "    Web UI  →  http://localhost:8000"
echo "    API     →  http://localhost:8000/docs"
echo "    Leads   →  http://localhost:8000/leads"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python server.py
