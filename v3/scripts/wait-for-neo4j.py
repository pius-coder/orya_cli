#!/usr/bin/env python3
"""Wait for Neo4j to be available before starting dependent services.

Fixes v2 issues:
- Uses parameterized Cypher for password change (no SQL injection)
- No hardcoded /app paths in sys.path
- Cleaner error handling
"""
import asyncio
import os
import sys
import time

from neo4j import AsyncGraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


def _uri() -> str:
    # Ensure we connect to localhost inside container
    return NEO4J_URI


async def check_neo4j() -> bool:
    uri = _uri()
    auth = (NEO4J_USER, NEO4J_PASSWORD)
    driver = AsyncGraphDatabase.driver(uri, auth=auth)
    try:
        await driver.verify_connectivity()
        await driver.close()
        return True
    except Exception as e:
        err_str = str(e).lower()
        if "unauthorized" in err_str or "credentials" in err_str:
            # Try default credentials and change password
            default_auth = (NEO4J_USER, "neo4j")
            driver2 = AsyncGraphDatabase.driver(uri, auth=default_auth)
            try:
                async with driver2.session() as session:
                    await session.run(
                        "ALTER CURRENT USER SET PASSWORD FROM $old TO $new",
                        old="neo4j",
                        new=NEO4J_PASSWORD,
                    )
                print("Neo4j password updated from default.")
                return True
            except Exception as e2:
                print(f"Neo4j default auth also failed: {e2}")
            finally:
                await driver2.close()
        return False
    finally:
        try:
            await driver.close()
        except Exception:
            pass


async def main() -> int:
    timeout = 120
    start = time.time()
    while time.time() - start < timeout:
        if await check_neo4j():
            print("Neo4j is ready.")
            return 0
        print("Waiting for Neo4j...")
        await asyncio.sleep(2)
    print("Neo4j did not become ready in time.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
