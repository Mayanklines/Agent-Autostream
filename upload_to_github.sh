#!/usr/bin/env bash
# =============================================================================
# upload_to_github.sh
# Initializes git repo and pushes project to GitHub.
#
# Usage:
#   chmod +x upload_to_github.sh
#   ./upload_to_github.sh
#
# Requirements:
#   - git installed
#   - GitHub CLI (gh) installed: https://cli.github.com
#     OR set GITHUB_TOKEN and GITHUB_USERNAME env vars for manual push
# =============================================================================

set -e

REPO_NAME="inflx-autostream-agent"
DESCRIPTION="AutoStream AI Sales Agent – LangGraph + Claude 3 Haiku conversational lead capture agent (Inflx / ServiceHive ML Intern Assignment)"

echo ""
echo "🚀  AutoStream Agent — GitHub Upload Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Initialize git (if not already)
if [ ! -d ".git" ]; then
    echo "📁  Initializing git repository..."
    git init
    echo "✅  Git initialized"
else
    echo "ℹ️   Git already initialized"
fi

# 2. Stage all files
echo "📦  Staging files..."
git add .
git status --short

# 3. Commit
echo ""
echo "💬  Creating initial commit..."
git commit -m "feat: initial AutoStream AI sales agent

- LangGraph state machine with 5 nodes
- Claude 3 Haiku via langchain-anthropic
- RAG pipeline over JSON knowledge base
- Intent classification (5 categories)
- Lead collection with field-by-field extraction
- mock_lead_capture() tool with JSON persistence
- WhatsApp webhook deployment guide in README"

echo "✅  Commit created"

# 4. Create GitHub repo and push (using GitHub CLI)
if command -v gh &>/dev/null; then
    echo ""
    echo "🐙  Creating GitHub repository via GitHub CLI..."
    gh repo create "$REPO_NAME" \
        --public \
        --description "$DESCRIPTION" \
        --source=. \
        --remote=origin \
        --push
    echo ""
    echo "✅  Repository created and pushed!"
    echo "🔗  URL: https://github.com/$(gh api user --jq .login)/$REPO_NAME"
else
    echo ""
    echo "⚠️   GitHub CLI (gh) not found."
    echo "   Option A — Install it: https://cli.github.com"
    echo ""
    echo "   Option B — Manual push:"
    echo "   1. Create a new repo at https://github.com/new"
    echo "      Name it: $REPO_NAME"
    echo "      Keep it empty (no README)"
    echo ""
    echo "   2. Run these commands:"
    echo "      git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
    echo "      git branch -M main"
    echo "      git push -u origin main"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉  Done!"
