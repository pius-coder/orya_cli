#!/bin/bash
set -e

PGDATA="${PGDATA:-/var/lib/postgresql/data}"

# Initialize PostgreSQL if needed
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "Initializing PostgreSQL..."
    su postgres -c "initdb -D $PGDATA"

    # Allow local connections
    cat <<EOF >> "$PGDATA/pg_hba.conf"
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
EOF

    echo "listen_addresses = '127.0.0.1'" >> "$PGDATA/postgresql.conf"
fi

# Start PostgreSQL temporarily to create role and database
su postgres -c "pg_ctl -D $PGDATA -l $PGDATA/logfile start"
sleep 2

# Create role and database if they don't exist
su postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='orya'\" | grep -q 1 || psql -c \"CREATE ROLE orya WITH LOGIN PASSWORD 'orya_secret_2026';\""
su postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='orya'\" | grep -q 1 || psql -c \"CREATE DATABASE orya OWNER orya;\""

# Apply schema
if [ -f /app/db/init.sql ]; then
    su postgres -c "psql -d orya -f /app/db/init.sql"
fi

# Stop temporary instance (supervisord will start it properly)
su postgres -c "pg_ctl -D $PGDATA stop"

# Start PostgreSQL in foreground for supervisord
exec su postgres -c "postgres -D $PGDATA"
