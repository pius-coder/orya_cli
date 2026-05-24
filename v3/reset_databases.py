"""Safe database reset utility.

Fixes v2 issues:
- NO hardcoded fallback IP for Neo4j (must be set in env)
- Interactive confirmation
- Resets reflections table too
"""
import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
POSTGRES_DSN = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")


def _require_env() -> None:
    missing = []
    if not NEO4J_URI:
        missing.append("NEO4J_URI")
    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")
    if not POSTGRES_DSN:
        missing.append("POSTGRES_DSN (or DATABASE_URL)")
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        sys.exit(1)


async def reset_postgres() -> None:
    pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=2)
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM orya.feedback")
        await conn.execute("DELETE FROM orya.opt_ins")
        await conn.execute("DELETE FROM orya.reflections")
        await conn.execute("DELETE FROM orya.sessions")
        await conn.execute("DELETE FROM orya.users")
    await pool.close()
    print("PostgreSQL reset complete.")


def reset_neo4j() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    print("Neo4j reset complete.")


async def main() -> None:
    _require_env()
    print("WARNING: This will DELETE ALL DATA in Neo4j and PostgreSQL.")
    print(f"Neo4j: {NEO4J_URI}")
    print(f"Postgres: {POSTGRES_DSN}")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    reset_neo4j()
    await reset_postgres()
    print("All databases reset.")


if __name__ == "__main__":
    asyncio.run(main())
