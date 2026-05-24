#!/bin/bash
set -e

# Dossier racine du serveur (là où se trouve ce script)
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "🚀 Démarrage de l'environnement Local (Sans Docker)"
echo "=========================================="

# ── Installation des dépendances Gateway ──────────────────────────
echo "📦 Installation des dépendances Gateway (Bun)..."
(cd "$DIR/gateway" && bun install)

# ── Lancement des services ────────────────────────────────────────
echo "🟢 Lancement de Graphiti-Server (Port 8000)..."
(cd "$DIR/graphiti-server" && uv run --python .venv/bin/python3 -- python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload) &
GRAPHITI_PID=$!

echo "🟢 Lancement de l'Agent (Port 5001)..."
(cd "$DIR" && uv run --python agent/.venv/bin/python3 -- python3 -m uvicorn agent.main:app --host 127.0.0.1 --port 5001 --reload) &
AGENT_PID=$!

echo "🟢 Lancement du Gateway (Port 4001)..."
(cd "$DIR/gateway" && bun run --hot src/index.ts) &
GATEWAY_PID=$!

echo "=========================================="
echo "✅ Tous les services sont lancés !"
echo "   Graphiti : http://localhost:8000/docs"
echo "   Agent    : http://localhost:5001/docs"
echo "   Gateway  : http://localhost:4001"
echo "Pour tout arrêter, appuyez sur Ctrl+C"
echo "=========================================="

# Fonction pour tuer proprement tous les processus enfants quand on fait Ctrl+C
cleanup() {
    echo ""
    echo "🛑 Arrêt des services..."
    kill $GRAPHITI_PID $AGENT_PID $GATEWAY_PID 2>/dev/null
    wait 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Attendre infiniment jusqu'à ce que l'utilisateur fasse Ctrl+C
wait

