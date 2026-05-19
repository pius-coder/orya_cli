#!/usr/bin/env python3
"""
Orya v2 — Simulation E2E avec 4 personnes.

Scénario:
  1. Marc (plombier à Lyon) — offre ses services
  2. Sophie (dev fullstack) — cherche un plombier
  3. Karim (ami, DJ le week-end) — propose ses services DJ
  4. Julie (organisatrice d'événements) — cherche un dev ET un DJ

Chaque personne discute naturellement avec Orya, puis cherche quelque chose.
On vérifie que le matching fonctionne (les extracted facts permettent le cross-search).
"""

import asyncio
import json
import sys
import time

import httpx

AGENT_URL = "http://127.0.0.1:5001"
GRAPHITI_URL = "http://127.0.0.1:8000"

# ── Les 4 personnages ─────────────────────────────────────────────

USERS = {
    "marc": {"alias": "Marc", "description": "plombier Lyon"},
    "sophie": {"alias": "Sophie", "description": "dev fullstack Paris"},
    "karim": {"alias": "Karim", "description": "ami DJ le week-end"},
    "julie": {"alias": "Julie", "description": "organisatrice d'événements"},
}

# ── Conversations (chaque user dit plusieurs choses) ──────────────

CONVERSATIONS = {
    "marc": [
        "salut ! moi c'est Marc, je suis plombier à Lyon depuis 8 ans",
        "ouais je fais tout : chaudières, fuites, installations neuves. J'ai ma propre boîte",
        "en ce moment je suis plutôt dispo, c'est la période creuse",
        "je cherche des clients dans le coin de Lyon 3e et Villeurbanne",
    ],
    "sophie": [
        "hey ! moi c'est Sophie, je suis développeuse fullstack",
        "je bosse en React + Node principalement, un peu de Python aussi",
        "là j'ai un souci de fuite dans ma cuisine, c'est la galère",
        "tu connais un plombier de confiance sur Lyon ? j'habite Lyon 3e",
    ],
    "karim": [
        "yo ! Karim ici, je suis comptable en semaine mais DJ le week-end",
        "je mixe de la house et de la techno, j'ai mon propre matos",
        "je suis dispo pour des soirées privées, mariages, tout ça",
        "je suis sur Paris et banlieue, je me déplace avec mon van",
    ],
    "julie": [
        "salut, moi c'est Julie ! j'organise des événements à Paris",
        "je prépare un mariage pour dans 2 mois et il me manque un DJ",
        "aussi j'ai besoin d'un dev pour créer un petit site pour l'événement",
        "le budget est correct, je cherche des gens fiables et dispos",
    ],
}

# ── Recherches (chacun cherche quelque chose de spécifique) ───────

SEARCHES = {
    "marc": "je cherche des gens qui ont besoin d'un plombier sur Lyon",
    "sophie": "tu as des contacts plombier sur Lyon 3e ?",
    "karim": "quelqu'un organise un event et cherche un DJ ?",
    "julie": "je cherche un DJ pour un mariage à Paris et un développeur web",
}


# ── Helpers ───────────────────────────────────────────────────────

async def chat(client: httpx.AsyncClient, user_id: str, alias: str, text: str) -> dict:
    """Envoie un message à l'agent et retourne la réponse complète."""
    resp = await client.post(
        f"{AGENT_URL}/chat",
        json={"user_id": user_id, "alias": alias, "text": text},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def print_response(user_id: str, text: str, response: dict):
    """Affiche joliment la réponse."""
    alias = USERS[user_id]["alias"]
    print(f"\n{'─'*60}")
    print(f"  👤 {alias}: {text}")
    print(f"  🤖 Orya: {response.get('reply', '(pas de réponse)')}")
    
    facts = response.get("facts", [])
    if facts:
        print(f"  📋 Facts extraits: {len(facts)}")
        for f in facts:
            print(f"     • {f['label']}: {f['value']} (conf: {f['confidence']:.0%})")
    
    candidates = response.get("candidates", [])
    if candidates:
        print(f"  🎯 CANDIDATS TROUVÉS: {len(candidates)}")
        for c in candidates:
            print(f"     ★ {c.get('alias', c['user_id'])} — {c['summary']} (score: {c['score']:.2f})")
    
    trace = response.get("trace", [])
    if trace:
        steps = [t["step"] for t in trace]
        print(f"  🔍 Trace: {' → '.join(steps)}")


async def run_simulation():
    """Exécute la simulation complète."""
    print("=" * 60)
    print("  ORYA v2 — SIMULATION E2E (4 personnes)")
    print("=" * 60)
    
    # Vérifier que l'agent est up
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{AGENT_URL}/health", timeout=5.0)
            h = health.json()
            print(f"\n✅ Agent health: {h}")
        except Exception as e:
            print(f"\n❌ Agent non disponible: {e}")
            sys.exit(1)

    async with httpx.AsyncClient() as client:
        # ── Phase 1: Conversations ────────────────────────────────
        print("\n\n" + "═" * 60)
        print("  PHASE 1 — CONVERSATIONS (chacun parle de soi)")
        print("═" * 60)
        
        for user_id, messages in CONVERSATIONS.items():
            alias = USERS[user_id]["alias"]
            print(f"\n{'━'*60}")
            print(f"  📱 Session: {alias} ({USERS[user_id]['description']})")
            print(f"{'━'*60}")
            
            for msg in messages:
                try:
                    response = await chat(client, user_id, alias, msg)
                    print_response(user_id, msg, response)
                except Exception as e:
                    print(f"\n  ❌ ERREUR pour {alias}: {e}")
                
                # Petit délai pour ne pas surcharger l'API Groq
                await asyncio.sleep(2)
        
        # ── Phase 2: Recherches ───────────────────────────────────
        print("\n\n" + "═" * 60)
        print("  PHASE 2 — RECHERCHES (chacun cherche quelque chose)")
        print("═" * 60)
        
        all_candidates = {}
        
        for user_id, search_text in SEARCHES.items():
            alias = USERS[user_id]["alias"]
            print(f"\n{'━'*60}")
            print(f"  🔎 {alias} cherche: {search_text}")
            print(f"{'━'*60}")
            
            try:
                response = await chat(client, user_id, alias, search_text)
                print_response(user_id, search_text, response)
                all_candidates[user_id] = response.get("candidates", [])
            except Exception as e:
                print(f"\n  ❌ ERREUR recherche {alias}: {e}")
                all_candidates[user_id] = []
            
            await asyncio.sleep(3)
        
        # ── Phase 3: Vérification du graphe ───────────────────────
        print("\n\n" + "═" * 60)
        print("  PHASE 3 — VÉRIFICATION GRAPHITI (cross-search)")
        print("═" * 60)
        
        # Recherche directe dans Graphiti pour confirmer les facts
        searches_graphiti = [
            ("plombier Lyon", "Devrait trouver Marc"),
            ("développeur fullstack React", "Devrait trouver Sophie"),
            ("DJ mariage Paris", "Devrait trouver Karim"),
            ("organisatrice événements", "Devrait trouver Julie"),
        ]
        
        for query, expected in searches_graphiti:
            try:
                resp = await client.post(
                    f"{GRAPHITI_URL}/retrieve/search",
                    json={"query": query, "num_results": 5},
                    timeout=30.0,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                print(f"\n  🔍 Graphiti search: '{query}'")
                print(f"     Attendu: {expected}")
                print(f"     Résultats: {len(results)} edges trouvés")
                for r in results[:3]:
                    print(f"       • {r.get('fact', '?')}")
            except Exception as e:
                print(f"\n  ❌ Graphiti search failed for '{query}': {e}")
            
            await asyncio.sleep(1)
        
        # ── Résumé final ──────────────────────────────────────────
        print("\n\n" + "═" * 60)
        print("  RÉSUMÉ DES MATCHINGS")
        print("═" * 60)
        
        expected_matches = {
            "sophie": ["marc"],       # Sophie cherche plombier → Marc
            "julie": ["karim", "sophie"],  # Julie cherche DJ → Karim, dev → Sophie
            "karim": ["julie"],       # Karim cherche event → Julie
            "marc": ["sophie"],       # Marc cherche clients plomberie → Sophie
        }
        
        total_expected = 0
        total_found = 0
        
        for user_id, expected_targets in expected_matches.items():
            alias = USERS[user_id]["alias"]
            candidates = all_candidates.get(user_id, [])
            found_ids = [c["user_id"] for c in candidates]
            
            print(f"\n  {alias}:")
            print(f"    Attendu: {[USERS[t]['alias'] for t in expected_targets]}")
            print(f"    Trouvé:  {[c.get('alias', c['user_id']) for c in candidates]}")
            
            for target in expected_targets:
                total_expected += 1
                if target in found_ids:
                    total_found += 1
                    print(f"    ✅ Match {USERS[target]['alias']} trouvé !")
                else:
                    print(f"    ⚠️  Match {USERS[target]['alias']} pas trouvé (peut nécessiter plus de contexte)")
        
        print(f"\n\n{'═'*60}")
        print(f"  SCORE: {total_found}/{total_expected} matchings trouvés")
        print(f"{'═'*60}")
        
        if total_found == total_expected:
            print("  🎉 SIMULATION PARFAITE — Tous les matchings fonctionnent !")
        elif total_found > 0:
            print("  ✅ SIMULATION PARTIELLE — Le système fonctionne, certains matchings")
            print("     nécessitent plus de conversation ou de temps d'indexation.")
        else:
            print("  ⚠️  Aucun matching trouvé — vérifier l'extraction de facts et le search.")
        
        print("\n")


if __name__ == "__main__":
    asyncio.run(run_simulation())
