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
        err_str = str(e)
        # Check if it's an authorization error
        if "unauthorized" in err_str.lower() or "unauthenticated" in err_str.lower() or "security" in err_str.lower():
            print("Configured password unauthorized. Trying default credentials 'neo4j/neo4j' to self-heal...", flush=True)
            default_driver = AsyncGraphDatabase.driver(uri, auth=("neo4j", "neo4j"))
            try:
                # Execute the password change command on the system database
                async with default_driver.session(database="system") as session:
                    await session.run(
                        f"ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO '{password}'"
                    )
                print(f"Self-healed: Changed Neo4j default password to configured password!", flush=True)
            except Exception as default_e:
                print(f"Could not change default password: {default_e}", flush=True)
            finally:
                await default_driver.close()
        else:
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
