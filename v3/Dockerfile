# Orya v3 — All-in-one Dockerfile (Ubuntu 24.04)
# Services: Neo4j, PostgreSQL, Graphiti-server, Agent, Gateway

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl wget ca-certificates gnupg \
    python3 python3-venv python3-pip \
    supervisor \
    postgresql-16 postgresql-client-16 \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# ── Bun ───────────────────────────────────────────────────────────
RUN curl -fsSL https://bun.sh/install | bash && \
    mv ~/.bun/bin/bun /usr/local/bin/bun

# ── Neo4j ─────────────────────────────────────────────────────────
RUN wget -qO - https://debian.neo4j.com/neotechnology.gpg.key | gpg --dearmor -o /usr/share/keyrings/neo4j.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 5' > /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && apt-get install -y neo4j=1:5.26.0 && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────
COPY agent/requirements.txt /tmp/agent-requirements.txt
COPY graphiti-server/requirements.txt /tmp/graphiti-requirements.txt
RUN pip install --break-system-packages --no-cache-dir \
    -r /tmp/agent-requirements.txt \
    -r /tmp/graphiti-requirements.txt

# ── App code ──────────────────────────────────────────────────────
WORKDIR /app
COPY agent/ ./agent/
COPY graphiti-server/ ./graphiti-server/
COPY gateway/ ./gateway/
COPY db/ ./db/
COPY scripts/ ./scripts/
COPY supervisord.conf ./

# ── Gateway deps ──────────────────────────────────────────────────
RUN cd /app/gateway && bun install

# ── Permissions ───────────────────────────────────────────────────
RUN chmod +x /app/scripts/*.sh && \
    mkdir -p /var/lib/postgresql/data /var/lib/neo4j/data && \
    chown -R postgres:postgres /var/lib/postgresql/data && \
    chown -R neo4j:neo4j /var/lib/neo4j/data

# ── Ports ─────────────────────────────────────────────────────────
EXPOSE 4001  # Gateway WebSocket
EXPOSE 5001  # Agent FastAPI
EXPOSE 8000  # Graphiti REST
EXPOSE 7474  # Neo4j Browser HTTP
EXPOSE 7687  # Neo4j Bolt

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["supervisord", "-c", "/app/supervisord.conf"]
