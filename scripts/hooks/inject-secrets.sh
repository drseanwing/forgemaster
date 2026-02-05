#!/usr/bin/env bash
# Forgemaster secrets injection hook for Claude Code.
#
# This script is a PreToolUse hook that intercepts Bash tool calls
# and injects environment variables for Docker Compose commands.
#
# Installation:
#   chmod 600 ~/.claude/hooks/inject-secrets.sh
#   Add to ~/.claude/settings.json:
#   { "hooks": { "PreToolUse": [{ "matcher": "Bash", "command": "path/to/inject-secrets.sh" }] } }

set -euo pipefail

# Read tool input from stdin
INPUT=$(cat)

# Extract the command being executed
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')

# Only intercept Bash tool calls
if [[ "$TOOL_NAME" != "Bash" ]]; then
    echo "$INPUT"
    exit 0
fi

COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Secret registry - EDIT THESE VALUES
# Store actual secrets here (file permissions: chmod 600)
declare -A SECRETS=(
    ["ANTHROPIC_API_KEY"]="${ANTHROPIC_API_KEY:-}"
    ["GITHUB_TOKEN"]="${GITHUB_TOKEN:-}"
    ["DOCKERHUB_USERNAME"]="${DOCKERHUB_USERNAME:-}"
    ["DOCKERHUB_TOKEN"]="${DOCKERHUB_TOKEN:-}"
    ["OPENAI_API_KEY"]="${OPENAI_API_KEY:-}"
    ["HOSTINGER_API_TOKEN"]="${HOSTINGER_API_TOKEN:-}"
)

# Check if command matches docker compose patterns
if echo "$COMMAND" | grep -qE '(docker compose|docker-compose)\s+(up|run|exec|build)'; then
    # Build env prefix
    ENV_PREFIX=""
    for key in "${!SECRETS[@]}"; do
        if [[ -n "${SECRETS[$key]}" ]]; then
            ENV_PREFIX="${ENV_PREFIX}${key}=${SECRETS[$key]} "
        fi
    done

    # Modify command with env vars prepended
    if [[ -n "$ENV_PREFIX" ]]; then
        MODIFIED_COMMAND="env ${ENV_PREFIX}${COMMAND}"
        echo "$INPUT" | jq --arg cmd "$MODIFIED_COMMAND" '.tool_input.command = $cmd'
    else
        echo "$INPUT"
    fi
else
    echo "$INPUT"
fi
