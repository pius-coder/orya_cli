"""Real tool implementations — async functions that call Graphiti and PostgreSQL.

These are the actual functions executed when the LLM decides to call a tool.
Each receives the user_id + arguments and returns a string result.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from graphiti_core import Graphiti

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Holds references to Graphiti and PG so tools can access them."""

    def __init__(self, graphiti: Graphiti):
        self.graphiti = graphiti

    # ── Tool 1: Search user memory ─────────────────────────────

    async def search_user_memory(self, user_id: str, query: str) -> str:
        """Search Graphiti for facts about this user."""
        try:
            edges = await self.graphiti.search(
                query=query,
                group_ids=[user_id],
                num_results=5,
            )
            facts = []
            for edge in edges or []:
                fact = getattr(edge, "fact", None)
                if fact:
                    facts.append(f"- {fact}")
            if facts:
                return f"Souvenirs trouvés :\n" + "\n".join(facts)
            return "Aucun souvenir trouvé pour cette recherche."
        except Exception as e:
            logger.warning("search_user_memory failed: %s", e)
            return f"Erreur de recherche mémoire: {e}"

    # ── Tool 2: Search providers (cross-group) ───────────────

    async def search_providers(self, query: str, location: str | None = None) -> str:
        """Search across all users for providers matching the query."""
        try:
            # Build enriched query
            enriched = query
            if location:
                enriched = f"{enriched} {location}"

            edges = await self.graphiti.search(
                query=enriched,
                num_results=10,
            )

            # Group by user (group_id)
            per_user: dict[str, list[str]] = {}
            for edge in edges or []:
                gid = getattr(edge, "group_id", None) or getattr(edge, "groupId", None)
                if not gid:
                    continue
                fact = getattr(edge, "fact", None)
                if fact:
                    per_user.setdefault(gid, []).append(fact)

            if not per_user:
                return "Aucun prestataire trouvé pour cette recherche."

            lines = ["Profils trouvés :"]
            for uid, facts in list(per_user.items())[:3]:
                lines.append(f"\n- Utilisateur {uid}:")
                for f in facts[:3]:
                    lines.append(f"  • {f}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("search_providers failed: %s", e)
            return f"Erreur recherche prestataires: {e}"

    # ── Tool 3: Get pending matchings ──────────────────────────

    async def get_pending_matchings(self, user_id: str) -> str:
        """Get opt-ins awaiting the user's decision."""
        try:
            from agent.db import list_pending_opt_ins
            opt_ins = await list_pending_opt_ins(user_id)
            if not opt_ins:
                return "Aucune mise en relation en attente."

            lines = ["Mises en relation en attente :"]
            for opt in opt_ins:
                lines.append(
                    f"- Prestataire '{opt['provider_id']}' pour : {opt['need_summary']} "
                    f"(ID: {opt['opt_in_id']})"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning("get_pending_matchings failed: %s", e)
            return f"Erreur récupération matchings: {e}"

    # ── Tool 4: Get user profile ───────────────────────────────

    async def get_user_profile(self, user_id: str) -> str:
        """Get user profile from PG."""
        try:
            from agent.db import get_user, get_reflections
            user_row = await get_user(user_id)
            user_ref, orya_ref = await get_reflections(user_id)

            lines = [f"Profil de {user_id} :"]
            if user_row:
                alias = user_row.get("alias")
                tutoyer = user_row.get("tutoyer", True)
                lines.append(f"- Alias : {alias or 'non défini'}")
                lines.append(f"- Tutoiement : {'oui' if tutoyer else 'non'}")
            if user_ref:
                lines.append(f"- Mémoire utilisateur : {user_ref[:200]}...")
            if orya_ref:
                lines.append(f"- Notes Orya : {orya_ref[:200]}...")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("get_user_profile failed: %s", e)
            return f"Erreur profil: {e}"

    # ── Tool 5: Record event ───────────────────────────────────

    async def record_event(self, user_id: str, event_type: str, description: str) -> str:
        """Record a significant event. In v3 this is a no-op placeholder
        (events are recorded automatically via Graphiti add_episode)."""
        return f"Événement noté : {event_type} — {description[:100]}"


# Mapping: tool name → executor method
TOOL_MAP: dict[str, str] = {
    "search_user_memory": "search_user_memory",
    "search_providers": "search_providers",
    "get_pending_matchings": "get_pending_matchings",
    "get_user_profile": "get_user_profile",
    "record_event": "record_event",
}
