# ── Orya Server — All-in-One ────────────────────────────────────────
# Un seul conteneur avec TOUT :
#   - FalkorDB (graph database)
#   - Gateway (Hono/Bun WebSocket)
#   - Agent Orya (Python/FastAPI)
#   - Orchestrator (Python/FastAPI)
#   - Memory Service (Python/FastAPI)
#
# Coolify:
#   - Dockerfile path: server/Dockerfile
#   - Port exposé: 4001
#   - Variables d'env: voir .env.example
#   - Volume: /data (FalkorDB persistence)
#
# Usage local:
#   docker build -t orya-server ./server
#   docker run -d --name orya -p 4001:4001 --env-file server/.env orya-server

# ── Stage 1: Grab FalkorDB module from official image ──────────────
FROM falkordb/falkordb:latest AS falkordb-source

# ── Stage 2: Main image ───────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    python3 \
    python3-venv \
    redis-server \
    redis-tools \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv (fast Python package manager) ──────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Copy FalkorDB module from stage 1 ─────────────────────────────
COPY --from=falkordb-source /var/lib/falkordb/bin/falkordb.so /opt/falkordb/falkordb.so

# ── Install Bun ───────────────────────────────────────────────────
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

# ── Setup Gateway (Hono/Bun) ─────────────────────────────────────
WORKDIR /app/gateway
COPY gateway/package.json gateway/bun.lock* ./
RUN bun install
COPY gateway/ .

# ── Setup Python services (using uv for speed) ────────────────────
WORKDIR /app/agent-orya
COPY services/agent-orya/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY services/agent-orya/ .

WORKDIR /app/orchestrator
COPY services/orchestrator/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY services/orchestrator/ .

WORKDIR /app/memory
COPY services/memory/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY services/memory/ .

# ── Data volume ───────────────────────────────────────────────────
RUN mkdir -p /data
VOLUME /data

# ── Supervisor config (lance tous les process) ────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/orya.conf

# ── Expose only the gateway port ─────────────────────────────────
EXPOSE 4001

# ── Health check ──────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:4001/health || exit 1

# ── Start everything via supervisor ──────────────────────────────
WORKDIR /app
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/orya.conf"]
