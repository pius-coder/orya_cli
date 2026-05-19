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

# ── Patch Graphiti for Groq + FalkorDB compatibility ──────────────
# 1. Reduce hardcoded max_tokens (Groq Llama 4 Scout limit = 8192)
# 2. Fix FalkorDB edge save: SET r = edge fails because fact_embedding
#    is a list[float] that FalkorDB rejects before vecf32() can convert it.
#    We must exclude fact_embedding from the bulk SET and only use vecf32().
COPY <<'EOF' /tmp/patch_graphiti.py
import re

# Patch 1: max_tokens
for path in [
    "/usr/local/lib/python3.12/dist-packages/graphiti_core/utils/maintenance/edge_operations.py",
]:
    with open(path) as f:
        content = f.read()
    content = content.replace("extract_edges_max_tokens = 16384", "extract_edges_max_tokens = 8000")
    with open(path, "w") as f:
        f.write(content)

path = "/usr/local/lib/python3.12/dist-packages/graphiti_core/llm_client/client.py"
with open(path) as f:
    content = f.read()
content = content.replace("DEFAULT_MAX_TOKENS = 16384", "DEFAULT_MAX_TOKENS = 8000")
with open(path, "w") as f:
    f.write(content)

# Patch 2: FalkorDB edge queries - exclude fact_embedding from SET r = edge
path = "/usr/local/lib/python3.12/dist-packages/graphiti_core/models/edges/edge_db_queries.py"
with open(path) as f:
    content = f.read()

# Fix single edge save for FalkorDB
old_single = '''MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e = $edge_data
                SET e.fact_embedding = vecf32($edge_data.fact_embedding)
                RETURN e.uuid AS uuid'''

new_single = '''MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e.uuid = $edge_data.uuid,
                    e.group_id = $edge_data.group_id,
                    e.name = $edge_data.name,
                    e.fact = $edge_data.fact,
                    e.episodes = $edge_data.episodes,
                    e.created_at = $edge_data.created_at,
                    e.expired_at = $edge_data.expired_at,
                    e.valid_at = $edge_data.valid_at,
                    e.invalid_at = $edge_data.invalid_at
                SET e.fact_embedding = vecf32($edge_data.fact_embedding)
                RETURN e.uuid AS uuid'''

content = content.replace(old_single, new_single)

# Fix bulk edge save for FalkorDB
old_bulk = '''UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
                SET r = edge
                SET r.fact_embedding = vecf32(edge.fact_embedding)
                WITH r, edge
                RETURN edge.uuid AS uuid'''

new_bulk = '''UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
                SET r.uuid = edge.uuid,
                    r.group_id = edge.group_id,
                    r.name = edge.name,
                    r.fact = edge.fact,
                    r.episodes = edge.episodes,
                    r.created_at = edge.created_at,
                    r.expired_at = edge.expired_at,
                    r.valid_at = edge.valid_at,
                    r.invalid_at = edge.invalid_at
                SET r.fact_embedding = vecf32(edge.fact_embedding)
                WITH r, edge
                RETURN edge.uuid AS uuid'''

content = content.replace(old_bulk, new_bulk)

with open(path, "w") as f:
    f.write(content)

print("Graphiti patches applied successfully.")
EOF
RUN python3 /tmp/patch_graphiti.py && rm /tmp/patch_graphiti.py

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
