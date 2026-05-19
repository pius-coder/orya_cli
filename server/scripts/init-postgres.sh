#!/usr/bin/env bash
# Idempotent PostgreSQL bootstrap + run.
# - Initializes the data dir on first boot
# - Creates the orya role + database
# - Applies /docker-entrypoint-initdb.d/init.sql once
# - Then execs the postgres process in the foreground (so supervisord can manage it)

set -euo pipefail

PGDATA="${PGDATA:-/var/lib/postgresql/data}"
INIT_SQL="/docker-entrypoint-initdb.d/init.sql"
ORYA_USER="${POSTGRES_USER:-orya}"
ORYA_PASSWORD="${POSTGRES_PASSWORD:-orya_secret_2026}"
ORYA_DB="${POSTGRES_DB:-orya}"

# Locate the postgres binary directory (Ubuntu 24.04 ships pg16 at /usr/lib/postgresql/16/bin)
PG_BIN=""
for candidate in /usr/lib/postgresql/*/bin; do
    if [ -x "${candidate}/postgres" ]; then
        PG_BIN="${candidate}"
        break
    fi
done
if [ -z "${PG_BIN}" ]; then
    echo "[init-postgres] postgres binary not found" >&2
    exit 1
fi

mkdir -p "${PGDATA}"
chown -R postgres:postgres "$(dirname "${PGDATA}")"
chmod 700 "${PGDATA}"

if [ ! -s "${PGDATA}/PG_VERSION" ]; then
    echo "[init-postgres] initializing data dir at ${PGDATA}"
    su postgres -c "${PG_BIN}/initdb -D ${PGDATA} --auth-local=trust --auth-host=md5 --encoding=UTF-8"

    cat <<EOF >> "${PGDATA}/pg_hba.conf"
# Allow internal apps to connect via TCP with password
host all all 127.0.0.1/32 md5
host all all ::1/128 md5
EOF
    cat <<EOF >> "${PGDATA}/postgresql.conf"
listen_addresses = '127.0.0.1'
port = 5432
EOF

    echo "[init-postgres] starting postgres (one-shot for init)"
    su postgres -c "${PG_BIN}/pg_ctl -D ${PGDATA} -w start"

    echo "[init-postgres] creating role and database"
    su postgres -c "psql -v ON_ERROR_STOP=1 -c \"CREATE ROLE ${ORYA_USER} WITH LOGIN SUPERUSER PASSWORD '${ORYA_PASSWORD}';\""
    su postgres -c "psql -v ON_ERROR_STOP=1 -c \"CREATE DATABASE ${ORYA_DB} OWNER ${ORYA_USER};\""

    if [ -f "${INIT_SQL}" ]; then
        echo "[init-postgres] applying ${INIT_SQL}"
        su postgres -c "psql -v ON_ERROR_STOP=1 -d ${ORYA_DB} -f ${INIT_SQL}"
    fi

    echo "[init-postgres] stopping initial postgres"
    su postgres -c "${PG_BIN}/pg_ctl -D ${PGDATA} -w stop"
fi

echo "[init-postgres] launching postgres in foreground"
exec su postgres -c "${PG_BIN}/postgres -D ${PGDATA}"
