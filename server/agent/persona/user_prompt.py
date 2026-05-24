"""Module: user_prompt.

Dynamically formats and loads the user prompt context under XML/Markdown structure,
containing date/time, user description, facts, short-term history, long-term history from Neo4j,
and Orya's morphological settings.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# Cache for Neo4j driver to avoid reconnecting constantly
_NEO4J_DRIVER = None

def get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        uri = os.environ.get("NEO4J_URI", "bolt://54.157.51.154:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "orya_neo4j_password_2026")
        try:
            _NEO4J_DRIVER = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logger.warning("Could not connect to Neo4j in user_prompt: %s", e)
    return _NEO4J_DRIVER


async def fetch_long_term_episodes(user_id: str, limit: int = 5) -> list[str]:
    """Retrieve episodic memories from Neo4j for this user_id."""
    driver = get_neo4j_driver()
    if not driver:
        return []
    
    episodes = []
    try:
        # Run synchronous session in a threadpool to avoid blocking async loop
        def run_query():
            with driver.session() as session:
                query = """
                MATCH (e:Episodic)
                WHERE e.group_id = $user_id OR e.groupId = $user_id
                RETURN e.content as content, e.created_at as created_at
                ORDER BY e.created_at DESC
                LIMIT $limit
                """
                return list(session.run(query, user_id=user_id, limit=limit))
        
        import asyncio
        loop = asyncio.get_event_loop()
        records = await loop.run_in_executor(None, run_query)
        
        for r in reversed(records):
            content = r["content"]
            if content:
                episodes.append(str(content).strip())
    except Exception as e:
        logger.warning("Failed to fetch long term episodes: %s", e)
    return episodes


def format_message(msg: AnyMessage) -> str:
    """Format single LangChain message as standard dialogue line."""
    if isinstance(msg, HumanMessage):
        return f"Humain : {msg.content}"
    elif isinstance(msg, AIMessage):
        return f"Assistant : {msg.content}"
    elif isinstance(msg, SystemMessage):
        return f"Système : {msg.content}"
    else:
        content = getattr(msg, "content", str(msg))
        return f"Message : {content}"


async def build_user_prompt(
    *,
    user_id: str,
    user_alias: str | None,
    last_user_text: str,
    facts_context: list[str],
    history: list[AnyMessage],
    tutoyer: bool = True,
) -> str:
    """Build the dynamic user context block formatted as XML/Markdown."""
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Fetch long-term context from Graphiti episodic nodes
    long_term_episodes = await fetch_long_term_episodes(user_id, limit=5)
    
    # 2. Extract facts and separate matchings
    facts = []
    matchings = []
    in_matching_section = False
    
    for f in facts_context:
        if "MATCHINGS EN ATTENTE" in f or "---" in f:
            in_matching_section = True
            continue
        if in_matching_section:
            if "Prestataire" in f:
                matchings.append(f)
        else:
            facts.append(f)

    # 3. Format morphology and Orya's state
    morphology = []
    if tutoyer:
        morphology.append("- Mode d'échange : TUTOIEMENT respectueux et familier style SMS/WhatsApp.")
        morphology.append("- Ton d'Orya : Proche, direct, utilise des émojis sobres (`🙂`, `🙏🏾`, `👍🏾`). Tutoiement obligatoire.")
    else:
        morphology.append("- Mode d'échange : VOUVOIEMENT poli et bienveillant.")
        morphology.append("- Ton d'Orya : Professionnel, simple mais direct. Vouvoiement obligatoire.")

    # 4. Format short-term history
    short_term_lines = [format_message(m) for m in history[-6:]] if history else ["(Aucun échange récent)"]

    # 5. Build final prompt structure using XML tags
    lines = [
        "<user_context>",
        f"<current_time>Date et heure : {current_time_str}</current_time>",
        "",
        "<user_profile>",
        f"- Alias : {user_alias or user_id}",
        f"- ID Utilisateur : {user_id}",
        "</user_profile>",
        "",
        "<morphology_settings>",
        "\n".join(morphology),
        "</morphology_settings>",
    ]

    # Graph facts
    lines.append("")
    lines.append("<graph_facts>")
    if facts:
        for f in facts:
            lines.append(f"- {f}")
    else:
        lines.append("Aucun fait sémantique particulier extrait du graphe pour le moment.")
    lines.append("</graph_facts>")

    # Active Matchings/Opportunities
    if matchings:
        lines.append("")
        lines.append("<active_matchings>")
        for m in matchings:
            lines.append(f"- {m}")
        lines.append("</active_matchings>")

    # Long-term episodic history
    lines.append("")
    lines.append("<long_term_history>")
    if long_term_episodes:
        for ep in long_term_episodes:
            lines.append(f"- {ep}")
    else:
        lines.append("Aucune discussion passée enregistrée dans le graphe à long terme.")
    lines.append("</long_term_history>")

    # Short-term recent history
    lines.append("")
    lines.append("<short_term_history>")
    for line in short_term_lines:
        lines.append(line)
    lines.append("</short_term_history>")

    # Current user message
    lines.append("")
    lines.append("<current_message>")
    lines.append(last_user_text)
    lines.append("</current_message>")
    
    lines.append("</user_context>")

    return "\n".join(lines)
