#!/usr/bin/env python3
"""End-to-End simulation script for Orya v3.

Simulates 4 characters interacting with the agent and verifies:
  - Entity extraction & fact persistence
  - Cross-user matching (Marc↔Sophie, Karim↔Julie)
  - Sequential reveal & double opt-in
  - Memory retrieval after ingestion
  - Graphiti temporal knowledge graph

Usage:
  python3 test_simulation.py                  # default http://127.0.0.1:5001
  AGENT_URL=http://10.0.1.5:5001 python3 test_simulation.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx

AGENT_URL = os.getenv("AGENT_URL", "http://127.0.0.1:5001")
GRAPHITI_URL = os.getenv("GRAPHITI_URL", "http://127.0.0.1:8000")
TIMEOUT = 90.0

# ── Characters ─────────────────────────────────────────────────────

CHARACTERS = [
    {
        "user_id": "marc_001",
        "alias": "Marc",
        "messages": [
            "Je suis plombier à Lyon avec 10 ans d'expérience",
            "Je cherche des clients dans le centre de Lyon",
        ],
    },
    {
        "user_id": "sophie_001",
        "alias": "Sophie",
        "messages": [
            "J'habite à Lyon et mon évier fuit depuis hier",
            "Tu connais un bon plombier ? J'ai vraiment besoin d'aide",
        ],
    },
    {
        "user_id": "karim_001",
        "alias": "Karim",
        "messages": [
            "Je suis développeur frontend React depuis 5 ans",
            "Je cherche un job remote pour l'étranger",
        ],
    },
    {
        "user_id": "julie_001",
        "alias": "Julie",
        "messages": [
            "Je cherche un développeur frontend pour ma startup",
            "On est à Paris mais full remote possible",
        ],
    },
]

# ── Expectations ────────────────────────────────────────────────────

# After Phase 1, these user pairs should have been matched
EXPECTED_MATCHES = {
    ("sophie_001", "marc_001"): "plombier",
    ("julie_001", "karim_001"): "développeur",
}

# After Phase 1, each user should have stored facts
EXPECTED_FACTS_PER_USER = {
    "marc_001": ["plombier", "Lyon"],
    "sophie_001": ["évier", "Lyon"],
    "karim_001": ["React", "remote"],
    "julie_001": ["startup", "remote"],
}


# ── Helpers ────────────────────────────────────────────────────────

class SimResults:
    """Collects all simulation results for assertions."""

    def __init__(self) -> None:
        self.health: dict = {}
        self.chat_responses: list[tuple[str, str, dict]] = []
        self.opt_in_responses: list[tuple[str, str, dict]] = []
        self.memory_responses: list[tuple[str, dict]] = []
        self.graphiti_results: list[dict] = []
        self.errors: list[str] = []
        self.passed: int = 0
        self.failed: int = 0

    def record_chat(self, user_id: str, text: str, resp: dict) -> None:
        self.chat_responses.append((user_id, text, resp))

    def record_error(self, user_id: str, text: str, err: str) -> None:
        self.errors.append(f"[{user_id}] {text}: {err}")

    def assert_(self, condition: bool, label: str, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  ✅ {label}")
        else:
            self.failed += 1
            suffix = f" — {detail}" if detail else ""
            print(f"  ❌ {label}{suffix}")

    def summary(self) -> None:
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"  Errors ({len(self.errors)}):")
            for e in self.errors[:5]:
                print(f"    {e}")
        print(f"{'='*60}")
        if self.failed > 0:
            sys.exit(1)


async def chat(
    client: httpx.AsyncClient,
    user_id: str,
    alias: str,
    text: str,
) -> dict:
    resp = await client.post(
        f"{AGENT_URL}/chat",
        json={"user_id": user_id, "alias": alias, "text": text},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def feedback(
    client: httpx.AsyncClient,
    user_id: str,
    user_input: str,
    orya_response: str,
    rating: str,
) -> dict:
    resp = await client.post(
        f"{AGENT_URL}/feedback",
        json={
            "user_id": user_id,
            "user_input": user_input,
            "orya_response": orya_response,
            "rating": rating,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def print_response(user_id: str, text: str, response: dict) -> None:
    reply = response.get("reply", "(no reply)")
    truncated = reply[:120] + "…" if len(reply) > 120 else reply
    print(f"\n[{user_id}] {text}")
    print(f"  🤖 {truncated}")
    if response.get("candidates"):
        for c in response["candidates"]:
            alias = c.get("alias", c.get("user_id", "?"))
            summary = c.get("summary", "")[:60]
            score = c.get("score", 0)
            print(f"  👤 Candidate: {alias} (score={score:.2f}) — {summary}")
    if response.get("pending_opt_in"):
        p = response["pending_opt_in"]
        print(f"  📩 Pending opt-in: {p}")
    if response.get("trace"):
        for t in response["trace"][-5:]:
            step = t.get("step", "?")
            detail = t.get("detail", "")[:80]
            print(f"     [{step}] {detail}")


# ── Phases ──────────────────────────────────────────────────────────


async def phase_health(client: httpx.AsyncClient, r: SimResults) -> bool:
    """Phase 0: Health check — agent + postgres + graphiti must be up."""
    print("\n═══ Phase 0: Health Check ═══")
    try:
        resp = await client.get(f"{AGENT_URL}/health", timeout=10.0)
        data = resp.json()
        r.health = data
        print(f"  Agent health: {json.dumps(data)}")
        r.assert_(data.get("ok") is True, "Agent is healthy")
        r.assert_(data.get("services", {}).get("postgres") is True, "PostgreSQL connected")
        r.assert_(data.get("services", {}).get("graphiti") is True, "Graphiti connected")
        return data.get("ok", False)
    except Exception as e:
        r.record_error("system", "health", str(e))
        r.assert_(False, "Agent reachable", str(e))
        return False


async def phase_conversations(client: httpx.AsyncClient, r: SimResults) -> None:
    """Phase 1: Each character sends their messages."""
    print("\n═══ Phase 1: Conversations ═══")
    for char in CHARACTERS:
        for msg in char["messages"]:
            try:
                resp = await chat(client, char["user_id"], char["alias"], msg)
                r.record_chat(char["user_id"], msg, resp)
                print_response(char["user_id"], msg, resp)

                # Verify basic response structure
                r.assert_("reply" in resp, f"[{char['user_id']}] Response has reply")
                r.assert_(len(resp.get("reply", "")) > 0, f"[{char['user_id']}] Reply non-empty")

            except Exception as e:
                r.record_error(char["user_id"], msg, str(e))
                r.assert_(False, f"[{char['user_id']}] Chat succeeded", str(e))

            await asyncio.sleep(2)


async def phase_matching(client: httpx.AsyncClient, r: SimResults) -> None:
    """Phase 2: Wait for background matching, then check candidates."""
    print("\n═══ Phase 2: Cross-User Matching ═══")
    print("  Waiting 8s for background matching tasks…")
    await asyncio.sleep(8)

    # Re-trigger match-seeking users
    seeking_users = [
        ("sophie_001", "Sophie", "J'ai besoin d'un plombier urgent à Lyon"),
        ("julie_001", "Julie", "Je veux trouver un développeur frontend"),
    ]
    for uid, alias, text in seeking_users:
        try:
            resp = await chat(client, uid, alias, text)
            r.record_chat(uid, text, resp)
            print_response(uid, text, resp)
            r.assert_(len(resp.get("candidates", [])) > 0, f"[{uid}] Got candidates", f"candidates={resp.get('candidates')}")
        except Exception as e:
            r.record_error(uid, text, str(e))
            r.assert_(False, f"[{uid}] Match query succeeded", str(e))
        await asyncio.sleep(2)


async def phase_opt_in(client: httpx.AsyncClient, r: SimResults) -> None:
    """Phase 3: Accept any pending opt-ins (double opt-in flow)."""
    print("\n═══ Phase 3: Double Opt-In ═══")

    for char in CHARACTERS:
        try:
            resp = await chat(client, char["user_id"], char["alias"], "matchings ?")
            r.record_chat(char["user_id"], "matchings ?", resp)
            print_response(char["user_id"], "matchings ?", resp)

            pending = resp.get("pending_opt_in")
            if pending and pending.get("opt_in_id"):
                opt_in_id = pending["opt_in_id"]
                print(f"  📩 [{char['user_id']}] Accepting opt-in {opt_in_id}")
                accept_resp = await chat(
                    client,
                    char["user_id"],
                    char["alias"],
                    "Oui, j'accepte !",
                )
                r.opt_in_responses.append((char["user_id"], opt_in_id, accept_resp))
                print_response(char["user_id"], "accept", accept_resp)
            else:
                print(f"  ℹ️  [{char['user_id']}] No pending opt-in")

        except Exception as e:
            r.record_error(char["user_id"], "opt-in", str(e))

        await asyncio.sleep(1)


async def phase_memory(client: httpx.AsyncClient, r: SimResults) -> None:
    """Phase 4: Verify memory retrieval — ask each user about their profile."""
    print("\n═══ Phase 4: Memory Retrieval ═══")

    memory_queries = [
        ("marc_001", "Marc", "Dis-moi ce que tu sais sur moi"),
        ("sophie_001", "Sophie", "Qu'est-ce que tu sais sur moi ?"),
        ("karim_001", "Karim", "Rappelle-moi mon profil"),
        ("julie_001", "Julie", "Tu te souviens de quoi sur moi ?"),
    ]

    for uid, alias, query in memory_queries:
        try:
            resp = await chat(client, uid, alias, query)
            r.memory_responses.append((uid, resp))
            print_response(uid, query, resp)
            reply = resp.get("reply", "").lower()

            # Check that the reply mentions expected facts
            expected_keywords = EXPECTED_FACTS_PER_USER.get(uid, [])
            for kw in expected_keywords:
                r.assert_(
                    kw.lower() in reply,
                    f"[{uid}] Memory mentions '{kw}'",
                    f"reply={reply[:100]}",
                )

        except Exception as e:
            r.record_error(uid, query, str(e))
            r.assert_(False, f"[{uid}] Memory query succeeded", str(e))

        await asyncio.sleep(1)


async def phase_graphiti(client: httpx.AsyncClient, r: SimResults) -> None:
    """Phase 5: Verify Graphiti temporal knowledge graph."""
    print("\n═══ Phase 5: Graphiti Search ═══")

    queries = [
        ("plombier Lyon", "plombier"),
        ("développeur React remote", "React"),
    ]

    for query, expected_keyword in queries:
        try:
            resp = await client.post(
                f"{GRAPHITI_URL}/retrieve/search",
                json={"query": query, "num_results": 10},
                timeout=30.0,
            )
            data = resp.json()
            results = data.get("results", [])
            r.graphiti_results.append({"query": query, "count": len(results)})

            found = any(
                expected_keyword.lower() in str(r_.get("fact", "")).lower()
                for r_ in results
            )
            r.assert_(
                found or len(results) > 0,
                f"Graphiti: '{query}' returns results",
                f"count={len(results)}",
            )
            print(f"  🔍 '{query}' → {len(results)} results")

        except Exception as e:
            r.record_error("graphiti", query, str(e))
            r.assert_(False, f"Graphiti: '{query}' succeeded", str(e))


async def phase_feedback(client: httpx.AsyncClient, r: SimResults) -> None:
    """Phase 6: Submit feedback for each user."""
    print("\n═══ Phase 6: Feedback ═══")

    for uid, alias, msg, resp_data in [
        (c["user_id"], c["alias"], c["messages"][-1], None)
        for c in CHARACTERS
    ]:
        # Find the last reply for this user
        last_reply = ""
        for r_uid, r_text, r_resp in r.chat_responses:
            if r_uid == uid:
                last_reply = r_resp.get("reply", "")

        try:
            fb = await feedback(client, uid, msg, last_reply, "good")
            r.assert_(fb.get("ok") is True, f"[{uid}] Feedback recorded")
            print(f"  👍 [{uid}] Feedback OK")
        except Exception as e:
            r.record_error(uid, "feedback", str(e))
            r.assert_(False, f"[{uid}] Feedback succeeded", str(e))


async def phase_final_assertions(client: httpx.AsyncClient, r: SimResults) -> None:
    """Phase 7: Final cross-cutting assertions."""
    print("\n═══ Phase 7: Final Assertions ═══")

    # 1. Every user got at least one reply
    users_with_replies = {uid for uid, _, resp in r.chat_responses if resp.get("reply")}
    for char in CHARACTERS:
        r.assert_(
            char["user_id"] in users_with_replies,
            f"[{char['user_id']}] Received at least one reply",
        )

    # 2. At least one candidate was produced
    all_candidates = []
    for uid, text, resp in r.chat_responses:
        all_candidates.extend(resp.get("candidates", []))
    r.assert_(len(all_candidates) > 0, "At least one candidate was produced", f"total={len(all_candidates)}")

    # 3. Expected match pairs appeared
    if all_candidates:
        for (seeker, provider), keyword in EXPECTED_MATCHES.items():
            found = any(
                c.get("user_id") == provider
                or keyword.lower() in c.get("summary", "").lower()
                for c in all_candidates
            )
            r.assert_(
                found,
                f"Match: {seeker} ↔ {provider} ({keyword})",
                f"candidates={[c.get('user_id') for c in all_candidates]}",
            )

    # 4. Trace events were recorded
    all_traces = []
    for _, _, resp in r.chat_responses:
        all_traces.extend(resp.get("trace", []))
    r.assert_(len(all_traces) > 0, "Trace events were recorded", f"total={len(all_traces)}")

    # 5. No unexpected errors
    r.assert_(len(r.errors) == 0, "No unexpected errors", f"errors={len(r.errors)}")


# ── Main ────────────────────────────────────────────────────────────


async def run_simulation() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║        Orya v3 — E2E Simulation + Assertions        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Agent:   {AGENT_URL}")
    print(f"  Graphiti: {GRAPHITI_URL}")
    print(f"  Users:   {', '.join(c['alias'] for c in CHARACTERS)}")

    r = SimResults()
    start = time.time()

    async with httpx.AsyncClient() as client:
        # Phase 0
        if not await phase_health(client, r):
            r.summary()
            return

        # Phase 1
        await phase_conversations(client, r)

        # Phase 2
        await phase_matching(client, r)

        # Phase 3
        await phase_opt_in(client, r)

        # Phase 4
        await phase_memory(client, r)

        # Phase 5
        await phase_graphiti(client, r)

        # Phase 6
        await phase_feedback(client, r)

        # Phase 7
        await phase_final_assertions(client, r)

    elapsed = time.time() - start
    print(f"\n  ⏱  Elapsed: {elapsed:.1f}s")
    r.summary()


if __name__ == "__main__":
    asyncio.run(run_simulation())
