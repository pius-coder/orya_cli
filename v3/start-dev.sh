#!/bin/bash
set -e

# Dev startup script for Orya v3 (local, no Docker).
# Assumes Python venvs and Bun are already installed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Orya v3 Dev Launcher ==="

# Graphiti server (port 8000)
if [ -d "$SCRIPT_DIR/graphiti-server/.venv" ]; then
    GRAPHITI_VENV="$SCRIPT_DIR/graphiti-server/.venv/bin/python3"
else
    GRAPHITI_VENV="python3"
fi
PYTHONPATH="$SCRIPT_DIR" uv run --python "$GRAPHITI_VENV" -m uvicorn graphiti-server.main:app --host 0.0.0.0 --port 8000 --reload &
GRAPHITI_PID=$!
echo "Graphiti server PID: $GRAPHITI_PID"

# Agent (port 5001)
if [ -d "$SCRIPT_DIR/agent/.venv" ]; then
    AGENT_VENV="$SCRIPT_DIR/agent/.venv/bin/python3"
else
    AGENT_VENV="python3"
fi
PYTHONPATH="$SCRIPT_DIR" uv run --python "$AGENT_VENV" -m uvicorn agent.main:app --host 0.0.0.0 --port 5001 --reload &
AGENT_PID=$!
echo "Agent PID: $AGENT_PID"

# Gateway (port 4001)
cd "$SCRIPT_DIR/gateway"
if [ ! -d "node_modules" ]; then
    bun install
fi
bun --hot src/index.ts &
GATEWAY_PID=$!
echo "Gateway PID: $GATEWAY_PID"

cd "$SCRIPT_DIR"

cleanup() {
    echo "Shutting down services..."
    kill $GRAPHITI_PID $AGENT_PID $GATEWAY_PID 2>/dev/null || true
    wait
    exit
}
trap cleanup SIGINT SIGTERM

echo "All services started. Press Ctrl+C to stop."
wait
