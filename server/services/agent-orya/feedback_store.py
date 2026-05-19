"""
Feedback Store — Système de notation bon/mauvais sans fine-tuning.

Fonctionnement :
1. Chaque réponse d'Orya est stockée temporairement avec son input
2. L'utilisateur peut noter "bon" ou "mauvais"
3. Si "mauvais" → on regénère une réponse (retry) et si celle-ci est "bon" → on sauvegarde
4. Les "bonnes" réponses deviennent des few-shot dynamiques pour le futur
5. Les "mauvaises" deviennent des exemples négatifs contextuels

Stockage : fichier JSON local pour le MVP (plus tard → Redis/DB)
"""

import json
import os
from pathlib import Path
from typing import Optional

STORE_PATH = Path(os.getenv("FEEDBACK_STORE_PATH", "/app/data/feedback.json"))


def _load_store() -> dict:
    if not STORE_PATH.exists():
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return {"good": [], "bad": []}
    try:
        return json.loads(STORE_PATH.read_text())
    except:
        return {"good": [], "bad": []}


def _save_store(store: dict):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2))


def record_feedback(
    user_input: str,
    orya_response: str,
    rating: str,  # "good" | "bad"
    user_id: str,
) -> None:
    """Record a feedback entry."""
    store = _load_store()
    entry = {
        "input": user_input,
        "response": orya_response,
        "user_id": user_id,
        "ts": __import__("time").time(),
    }
    if rating == "good":
        store["good"].append(entry)
        # Keep max 100 good examples
        store["good"] = store["good"][-100:]
    else:
        store["bad"].append(entry)
        store["bad"] = store["bad"][-50:]
    _save_store(store)


def get_good_examples(n: int = 3, exclude_user: Optional[str] = None) -> list[dict]:
    """
    Get N random good examples for few-shot injection.
    Optionally exclude examples from the same user (to avoid echo).
    """
    import random
    store = _load_store()
    good = store.get("good", [])
    if exclude_user:
        good = [e for e in good if e.get("user_id") != exclude_user]
    if not good:
        return []
    sample = random.sample(good, min(n, len(good)))
    return [{"input": e["input"], "good": e["response"]} for e in sample]


def get_last_pending(user_id: str) -> Optional[dict]:
    """Get the last response that hasn't been rated yet (for retry flow)."""
    # Pending responses stored in memory for now (MVP simplification)
    # In production, use Redis with TTL
    return None
