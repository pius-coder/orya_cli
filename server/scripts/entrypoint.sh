#!/bin/bash
set -e

# Fix PostgreSQL directory permissions
chown -R postgres:postgres /var/lib/postgresql
chmod 700 /var/lib/postgresql/data || true

# Fix Neo4j directory permissions
chown -R neo4j:adm /var/lib/neo4j
chown -R neo4j:adm /usr/share/neo4j || true

# Execute supervisord
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/orya.conf
