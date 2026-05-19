# ── Orya v2 — All-in-One server image ──────────────────────────────
# Single container running:
#   - FalkorDB (graph DB, port 6379)
#   - PostgreSQL (relational, port 5432)
#   - Graphiti REST server (Python/FastAPI, port 8000)
#   - Agent (Python/FastAPI + LangGraph, port 5001)
#   - Gateway (Hono/Bun WebSocket, port 4001)
#
# Coolify:
#   - Dockerfile path: server/Dockerfile
#   - Exposed port: 4001 (and optionally 8000 for Graphiti REST inspection)
#   - Required env: see server/.env.example
#   - Volume: /data (FalkorDB) + /var/lib/postgresql/data (Postgres)

# ── Stage 1: pull FalkorDB binaries from the official image ──────
FROM falkordb/falkordb:latest AS falkordb-source

# ── Stage 2: main image ─────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System packages: Python 3.12, Postgres 16, Redis tools, Bun deps, supervisor.
RUN apt-get update && apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        libgomp1 \
        postgresql \
        postgresql-contrib \
        python3 \
        python3-venv \
        redis-tools \
        supervisor \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Bun (Gateway runtime)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

# FalkorDB module + redis-server + redis-cli from the official image
COPY --from=falkordb-source /var/lib/falkordb/bin/falkordb.so /opt/falkordb/falkordb.so
COPY --from=falkordb-source /usr/local/bin/redis-server /usr/local/bin/redis-server
COPY --from=falkordb-source /usr/local/bin/redis-cli /usr/local/bin/redis-cli

# ── PostgreSQL setup (init runs on first boot via init-postgres.sh) ──
RUN mkdir -p /var/lib/postgresql/data \
    && chown -R postgres:postgres /var/lib/postgresql \
    && chmod 700 /var/lib/postgresql/data
COPY db/init.sql /docker-entrypoint-initdb.d/init.sql
COPY scripts/init-postgres.sh /usr/local/bin/init-postgres.sh
RUN chmod +x /usr/local/bin/init-postgres.sh

# ── Agent (LangGraph) ────────────────────────────────────────────
WORKDIR /app/agent
COPY agent/requirements.txt ./requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt
COPY agent/ ./

# ── Graphiti REST server ─────────────────────────────────────────
WORKDIR /app/graphiti-server
COPY graphiti-server/requirements.txt ./requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt
COPY graphiti-server/ ./

# ── Gateway (Hono/Bun) ───────────────────────────────────────────
WORKDIR /app/gateway
COPY gateway/package.json gateway/tsconfig.json gateway/bun.lock* ./
RUN bun install
COPY gateway/ ./

# ── Persistent volumes ───────────────────────────────────────────
RUN mkdir -p /data
VOLUME /data /var/lib/postgresql/data

# ── Supervisor config ────────────────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/orya.conf

# ── Run from /app so `agent.main:app` resolves as a package ──────
WORKDIR /app

EXPOSE 4001 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:4001/health || exit 1

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/orya.conf"]
