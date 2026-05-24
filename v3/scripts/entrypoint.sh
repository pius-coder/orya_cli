#!/bin/bash
set -e

# Entrypoint for Orya v3 all-in-one container.
# Idempotent setup then delegates to supervisord or custom command.

# Neo4j listen on all interfaces
NEO4J_CONF="/etc/neo4j/neo4j.conf"
if [ -f "$NEO4J_CONF" ]; then
    if ! grep -q "server.default_listen_address=0.0.0.0" "$NEO4J_CONF"; then
        echo "server.default_listen_address=0.0.0.0" >> "$NEO4J_CONF"
    fi
fi

# Ensure data dirs exist
mkdir -p /var/lib/postgresql/data
chmod 700 /var/lib/postgresql/data || true
mkdir -p /var/lib/neo4j/data
chown -R neo4j:neo4j /var/lib/neo4j/data || true

exec "$@"
