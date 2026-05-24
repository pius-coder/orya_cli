import asyncio
import os
import sys
from dotenv import load_dotenv

# Add agent path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

load_dotenv()

from agent.settings import get_settings
import asyncpg
from neo4j import GraphDatabase

async def reset_postgres():
    s = get_settings()
    print(f"Connecting to Postgres: {s.POSTGRES_HOST}:{s.POSTGRES_PORT} (DB: {s.POSTGRES_DB})")
    conn = await asyncpg.connect(
        host=s.POSTGRES_HOST,
        port=s.POSTGRES_PORT,
        user=s.POSTGRES_USER,
        password=s.POSTGRES_PASSWORD,
        database=s.POSTGRES_DB
    )
    try:
        # Delete ALL opt-ins
        res1 = await conn.execute("DELETE FROM orya.opt_ins")
        print(f"Deleted all opt_ins: {res1}")
        
        # Delete ALL feedback
        res2 = await conn.execute("DELETE FROM orya.feedback")
        print(f"Deleted all feedback: {res2}")
        
        # Delete ALL users
        res3 = await conn.execute("DELETE FROM orya.users")
        print(f"Deleted all users: {res3}")
        
        # Re-insert default system user
        await conn.execute(
            "INSERT INTO orya.users (user_id, alias) VALUES ('orya_default', 'Test User') ON CONFLICT DO NOTHING"
        )
        print("Re-inserted orya_default.")
        
    finally:
        await conn.close()

def reset_neo4j():
    s = get_settings()
    uri = os.environ.get("NEO4J_URI", "bolt://54.157.51.154:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "orya_neo4j_password_2026")
    
    print(f"Connecting to Neo4j to clear everything: {uri}")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Delete ALL nodes and relationships in the database
            query = "MATCH (n) DETACH DELETE n"
            result = session.run(query)
            summary = result.consume()
            print(f"Deleted from Neo4j: {summary.counters.nodes_deleted} nodes, {summary.counters.relationships_deleted} relationships.")
        driver.close()
    except Exception as e:
        print(f"Error resetting Neo4j: {e}")

async def main():
    print("=== STARTING DATABASES RESET FOR SIMULATION USERS ===")
    await reset_postgres()
    reset_neo4j()
    print("=== RESET COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(main())
