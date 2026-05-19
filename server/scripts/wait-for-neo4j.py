import asyncio
import os
import sys
import time
from neo4j import AsyncGraphDatabase

# Add graphiti-server and agent to path to load settings
sys.path.insert(0, "/app/graphiti-server")
from settings import get_settings

s = get_settings()
uri = s.NEO4J_URI
user = s.NEO4J_USER
password = s.NEO4J_PASSWORD

print(f"Waiting for Neo4j at {uri} with user {user}...", flush=True)

async def check_neo4j():
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        # Try to execute a simple query to verify auth and availability
        await driver.execute_query("RETURN 1")
        print("Neo4j is UP and authenticated!", flush=True)
        return True
    except Exception as e:
        print(f"Neo4j not ready yet: {e}", flush=True)
        return False
    finally:
        await driver.close()

async def main():
    start_time = time.time()
    while time.time() - start_time < 120:  # Timeout after 2 minutes
        if await check_neo4j():
            sys.exit(0)
        await asyncio.sleep(2)
    print("Timeout waiting for Neo4j", flush=True)
    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
