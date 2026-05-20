#!/bin/bash
set -e

# Fix PostgreSQL directory permissions
chown -R postgres:postgres /var/lib/postgresql
chmod 700 /var/lib/postgresql/data || true

# Fix Neo4j directory permissions
chown -R neo4j:adm /var/lib/neo4j
chown -R neo4j:adm /usr/share/neo4j || true

# Ensure Neo4j listens on all interfaces (0.0.0.0) instead of localhost
echo "server.default_listen_address=0.0.0.0" >> /etc/neo4j/neo4j.conf

# Execute command if provided (useful for decentralized docker-compose)
if [ $# -gt 0 ]; then
    exec "$@"
else
    # Default behavior: run everything via supervisord
    exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/orya.conf
fi
