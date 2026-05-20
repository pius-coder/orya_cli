"""Tool executor implementing the actual logic behind AGENT_TOOLS.

Fixes v2 issues:
- Uses static imports (no dynamic imports inside methods)
- Uses TOOL_MAP for dispatch (no manual if/elif in tool_agent.py)
- Better error messages
"""
from typing import Any

from graphiti_core import Graphiti

from ..db import get_good_examples, get_reflections, list_pending_opt_ins
from ..db.postgres import get_pool


class ToolExecutor:
    """Executes tool calls from the ReAct agent."""

    TOOL_MAP: dict[str, str] = {
        "search_user_memory": "search_user_memory",
        "search_providers": "search_providers",
        "get_pending_matchings": "get_pending_matchings",
        "get_user_profile": "get_user_profile",
        "record_event": "record_event",
    }

    def __init__(self, graphiti: Graphiti) -> None:
        self.graphiti = graphiti

    async def execute(self, tool_name: str, user_id: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call to the appropriate handler."""
        mapped = self.TOOL_MAP.get(tool_name)
        if not mapped:
            return f"[Erreur: outil inconnu {tool_name}]"

        handler = getattr(self, mapped, None)
        if not handler:
            return f"[Erreur: handler manquant pour {tool_name}]"

        try:
            return await handler(user_id=user_id, **args)
        except Exception as e:
            return f"[Erreur outil {tool_name}: {e}]"

    async def search_user_memory(self, user_id: str, query: str) -> str:
        results = await self.graphiti.search(
            query=query,
            group_ids=[user_id],
            num_results=5,
        )
        if not results:
            return "Je ne me souviens de rien à ce sujet."
        facts = []
        for r in results:
            fact = getattr(r, "fact", None) or str(r)
            facts.append(f"- {fact}")
        return "Faits trouvés :\n" + "\n".join(facts)

    async def search_providers(self, user_id: str, query: str, location: str | None = None) -> str:
        enriched = query
        if location:
            enriched = f"{query} à {location}"

        results = await self.graphiti.search(
            query=enriched,
            num_results=20,
        )
        if not results:
            return "Je n'ai trouvé personne pour l'instant."

        # Aggregate by provider (group_id)
        per_user: dict[str, list[str]] = {}
        for rank, r in enumerate(results, start=1):
            gid = getattr(r, "group_id", None)
            if not gid or gid == user_id:
                continue
            fact = getattr(r, "fact", None) or str(r)
            per_user.setdefault(gid, []).append(f"({rank}) {fact}")

        if not per_user:
            return "Je n'ai trouvé personne qui corresponde."

        # Build summary
        lines = []
        for uid, facts in sorted(per_user.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            alias = uid[:8]  # Will be resolved by matching layer
            top_facts = facts[:3]
            lines.append(f"- {alias} : " + " ; ".join(top_facts))

        return "Candidats trouvés :\n" + "\n".join(lines)

    async def get_pending_matchings(self, user_id: str) -> str:
        rows = await list_pending_opt_ins(user_id)
        if not rows:
            return "Aucun matching en attente."
        lines = []
        for r in rows:
            status = r.get("status", "?")
            reason = (r.get("reason") or "")[:80]
            lines.append(f"- {status}: {reason}")
        return "Matchings en attente :\n" + "\n".join(lines)

    async def get_user_profile(self, user_id: str) -> str:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT alias, tutoyer FROM orya.users WHERE id = $1", user_id
            )
        refs = await get_reflections(user_id)
        parts = []
        if row:
            parts.append(f"Alias: {row['alias']}")
            parts.append(f"Tutoiement: {row['tutoyer']}")
        if refs.get("user_reflection"):
            parts.append(f"Portrait: {refs['user_reflection'][:200]}")
        if refs.get("orya_reflection"):
            parts.append(f"Relation: {refs['orya_reflection'][:200]}")
        if not parts:
            return "Profil vide."
        return "Profil utilisateur :\n" + "\n".join(parts)

    async def record_event(self, user_id: str, event_type: str, description: str) -> str:
        # Placeholder: in future this could write to a dedicated events table
        return f"Événement '{event_type}' enregistré."
