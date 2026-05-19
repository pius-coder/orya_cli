# ── Orya v2 — All-in-One server image ──────────────────────────────
# Single container running:
#   - Neo4j Community Server (graph DB, port 7687 Bolt, port 7474 HTTP)
#   - PostgreSQL (relational, port 5432)
#   - Graphiti REST server (Python/FastAPI, port 8000)
#   - Agent (Python/FastAPI + LangGraph, port 5001)
#   - Gateway (Hono/Bun WebSocket, port 4001)
#
# Coolify:
#   - Dockerfile path: server/Dockerfile
#   - Exposed ports: 4001 (Gateway), 8000 (Graphiti), 7474 (Neo4j Browser), 7687 (Bolt)
#   - Required env: see server/.env.example
#   - Volume: /var/lib/neo4j/data + /var/lib/postgresql/data

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System packages: Java (required by Neo4j), Python 3.12, Postgres 16, Bun deps, supervisor.
RUN apt-get update && apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        libgomp1 \
        postgresql \
        postgresql-contrib \
        python3 \
        python3-venv \
        supervisor \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Install Neo4j Community Server
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | gpg --dearmor -o /usr/share/keyrings/neo4j.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 5" > /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && apt-get install -y openjdk-17-jre-headless neo4j && \
    rm -rf /var/lib/apt/lists/*

# Configure Neo4j: disable authentication and listen on all interfaces
RUN echo "dbms.security.auth_enabled=false" >> /etc/neo4j/neo4j.conf && \
    echo "server.default_listen_address=0.0.0.0" >> /etc/neo4j/neo4j.conf && \
    echo "server.bolt.listen_address=0.0.0.0:7687" >> /etc/neo4j/neo4j.conf && \
    echo "server.http.listen_address=0.0.0.0:7474" >> /etc/neo4j/neo4j.conf

# uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Bun (Gateway runtime)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

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
VOLUME /data /var/lib/postgresql/data /var/lib/neo4j/data

# ── Supervisor config ────────────────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/orya.conf

# ── Run from /app so `agent.main:app` resolves as a package ──────
WORKDIR /app

EXPOSE 4001 8000 7474 7687

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:4001/health || exit 1

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/orya.conf"]
