"""End-to-End simulation script for Orya v3.

Simulates 4 characters interacting with the agent and verifies cross-user matching.
"""
import asyncio
import json
import sys

import httpx

AGENT_URL = "http://127.0.0.1:5001"
GRAPHITI_URL = "http://127.0.0.1:8000"

CHARACTERS = [
    {"user_id": "marc_001", "alias": "Marc", "messages": [
        "Je suis plombier à Lyon",
        "Je cherche des clients dans le centre",
    ]},
    {"user_id": "sophie_001", "alias": "Sophie", "messages": [
        "J'habite à Lyon et mon évier fuit",
        "Tu connais un bon plombier ?",
    ]},
    {"user_id": "karim_001", "alias": "Karim", "messages": [
        "Je suis dev frontend React",
        "Je cherche un job remote",
    ]},
    {"user_id": "julie_001", "alias": "Julie", "messages": [
        "Je cherche un dev frontend pour mon startup",
        "On est à Paris mais remote OK",
    ]},
]


async def chat(client: httpx.AsyncClient, user_id: str, alias: str, text: str) -> dict:
    resp = await client.post(
        f"{AGENT_URL}/chat",
        json={"user_id": user_id, "alias": alias, "text": text},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def print_response(user_id: str, text: str, response: dict) -> None:
    print(f"\n[{user_id}] {text}")
    print(f"  -> {response.get('reply', '(no reply)')}")
    if response.get("candidates"):
        print(f"  -> candidates: {len(response['candidates'])}")
    if response.get("trace"):
        for t in response["trace"][-3:]:
            print(f"     [{t.get('step')}] {t.get('detail')}")


async def run_simulation() -> None:
    print("=== Orya v3 E2E Simulation ===")
    async with httpx.AsyncClient() as client:
        # Health check
        try:
            health = await client.get(f"{AGENT_URL}/health")
            print(f"Agent health: {health.json()}")
        except Exception as e:
            print(f"Agent not reachable: {e}")
            return

        # Phase 1: Conversations
        print("\n--- Phase 1: Conversations ---")
        for char in CHARACTERS:
            for msg in char["messages"]:
                try:
                    resp = await chat(client, char["user_id"], char["alias"], msg)
                    print_response(char["user_id"], msg, resp)
                except Exception as e:
                    print(f"ERROR [{char['user_id']}]: {e}")
                await asyncio.sleep(1)

        # Phase 2: Wait for background tasks
        print("\n--- Phase 2: Waiting for background matching ---")
        await asyncio.sleep(6)

        # Phase 3: Check opt-ins
        print("\n--- Phase 3: Checking opt-ins ---")
        for char in CHARACTERS:
            try:
                resp = await chat(client, char["user_id"], char["alias"], "matchings ?")
                print_response(char["user_id"], "matchings ?", resp)
            except Exception as e:
                print(f"ERROR [{char['user_id']}]: {e}")

        # Phase 4: Graphiti search
        print("\n--- Phase 4: Graphiti search ---")
        try:
            search_resp = await client.post(
                f"{GRAPHITI_URL}/retrieve/search",
                json={"query": "plombier Lyon", "num_results": 10},
                timeout=30.0,
            )
            print(f"Search results: {len(search_resp.json().get('results', []))}")
        except Exception as e:
            print(f"Graphiti search error: {e}")

    print("\n=== Simulation complete ===")


if __name__ == "__main__":
    asyncio.run(run_simulation())
