"""
Graph Store — FalkorDB interface for Orya's knowledge graph.

Schema:
  (User {id, alias, bio, isProvider, city})
  (Skill {name})
  (City {name})
  (Fact {kind, value, confidence, ts})

Relationships:
  (User)-[:HAS_SKILL]->(Skill)
  (User)-[:LIVES_IN]->(City)
  (User)-[:HAS_FACT]->(Fact)
  (User)-[:PROVIDES]->(Skill)  # for providers specifically
"""

import os
from typing import Optional

from falkordb import FalkorDB


class GraphStore:
    def __init__(self):
        self.host = os.getenv("FALKORDB_HOST", "localhost")
        self.port = int(os.getenv("FALKORDB_PORT", "6379"))
        self.graph_name = "orya"
        self.db: Optional[FalkorDB] = None
        self.graph = None

    async def connect(self):
        """Connect to FalkorDB."""
        try:
            self.db = FalkorDB(host=self.host, port=self.port)
            self.graph = self.db.select_graph(self.graph_name)
            # Create indexes
            await self._init_schema()
            print(f"[graph_store] connected to FalkorDB at {self.host}:{self.port}")
        except Exception as e:
            print(f"[graph_store] connection failed: {e} — running in degraded mode")
            self.graph = None

    async def _init_schema(self):
        """Create indexes and constraints."""
        if not self.graph:
            return
        try:
            self.graph.query("CREATE INDEX FOR (u:User) ON (u.id)")
            self.graph.query("CREATE INDEX FOR (s:Skill) ON (s.name)")
            self.graph.query("CREATE INDEX FOR (c:City) ON (c.name)")
        except:
            pass  # Indexes may already exist

    async def upsert_user(
        self,
        user_id: str,
        alias: str = "",
        skills: list[str] = [],
        city: str = "",
        bio: str = "",
        is_provider: bool = False,
    ):
        """Create or update a user node with relationships."""
        if not self.graph:
            return

        # Upsert user node
        self.graph.query(
            """
            MERGE (u:User {id: $id})
            SET u.alias = $alias, u.bio = $bio, u.isProvider = $isProvider
            """,
            {"id": user_id, "alias": alias, "bio": bio, "isProvider": is_provider},
        )

        # Link skills
        for skill in skills:
            skill_lower = skill.lower().strip()
            self.graph.query(
                """
                MERGE (s:Skill {name: $skill})
                MERGE (u:User {id: $uid})
                MERGE (u)-[:HAS_SKILL]->(s)
                """,
                {"skill": skill_lower, "uid": user_id},
            )
            if is_provider:
                self.graph.query(
                    """
                    MERGE (s:Skill {name: $skill})
                    MERGE (u:User {id: $uid})
                    MERGE (u)-[:PROVIDES]->(s)
                    """,
                    {"skill": skill_lower, "uid": user_id},
                )

        # Link city
        if city:
            self.graph.query(
                """
                MERGE (c:City {name: $city})
                MERGE (u:User {id: $uid})
                MERGE (u)-[:LIVES_IN]->(c)
                """,
                {"city": city.lower().strip(), "uid": user_id},
            )

    async def add_fact(self, user_id: str, kind: str, value: str, confidence: float):
        """Add a fact node linked to a user."""
        if not self.graph:
            return

        import time
        ts = time.time()

        # Create fact node and link to user
        self.graph.query(
            """
            MERGE (u:User {id: $uid})
            CREATE (f:Fact {kind: $kind, value: $value, confidence: $confidence, ts: $ts})
            CREATE (u)-[:HAS_FACT]->(f)
            """,
            {"uid": user_id, "kind": kind, "value": value, "confidence": confidence, "ts": ts},
        )

        # Special handling: if fact is a skill, also create Skill node
        if kind == "skill":
            self.graph.query(
                """
                MERGE (s:Skill {name: $skill})
                MERGE (u:User {id: $uid})
                MERGE (u)-[:HAS_SKILL]->(s)
                """,
                {"skill": value.lower().strip(), "uid": user_id},
            )

        # If fact is a city, link
        if kind == "city":
            self.graph.query(
                """
                MERGE (c:City {name: $city})
                MERGE (u:User {id: $uid})
                MERGE (u)-[:LIVES_IN]->(c)
                """,
                {"city": value.lower().strip(), "uid": user_id},
            )

    async def get_user_facts(self, user_id: str) -> list[str]:
        """Get all facts for a user as human-readable strings."""
        if not self.graph:
            return []

        try:
            result = self.graph.query(
                """
                MATCH (u:User {id: $uid})-[:HAS_FACT]->(f:Fact)
                RETURN f.kind, f.value
                ORDER BY f.ts DESC
                LIMIT 20
                """,
                {"uid": user_id},
            )
            facts = []
            for row in result.result_set:
                facts.append(f"{row[0]}: {row[1]}")

            # Also get skills and city
            skills_result = self.graph.query(
                """
                MATCH (u:User {id: $uid})-[:HAS_SKILL]->(s:Skill)
                RETURN s.name
                """,
                {"uid": user_id},
            )
            for row in skills_result.result_set:
                fact = f"skill: {row[0]}"
                if fact not in facts:
                    facts.append(fact)

            city_result = self.graph.query(
                """
                MATCH (u:User {id: $uid})-[:LIVES_IN]->(c:City)
                RETURN c.name
                LIMIT 1
                """,
                {"uid": user_id},
            )
            for row in city_result.result_set:
                fact = f"ville: {row[0]}"
                if fact not in facts:
                    facts.append(fact)

            return facts
        except Exception as e:
            print(f"[graph_store] get_user_facts error: {e}")
            return []

    async def search_providers(
        self,
        skills: list[str],
        city: Optional[str] = None,
        exclude_user: str = "",
        limit: int = 10,
    ) -> list[dict]:
        """
        Search the graph for users who PROVIDE matching skills.
        Optionally filter by city.
        """
        if not self.graph:
            return []

        try:
            if city:
                result = self.graph.query(
                    """
                    MATCH (u:User)-[:HAS_SKILL]->(s:Skill)
                    WHERE s.name IN $skills AND u.id <> $exclude
                    OPTIONAL MATCH (u)-[:LIVES_IN]->(c:City)
                    WHERE c.name = $city
                    RETURN u.id, u.alias, u.bio, collect(DISTINCT s.name) AS skills, c.name AS city
                    ORDER BY size(collect(DISTINCT s.name)) DESC
                    LIMIT $limit
                    """,
                    {
                        "skills": [s.lower().strip() for s in skills],
                        "city": city.lower().strip(),
                        "exclude": exclude_user,
                        "limit": limit,
                    },
                )
            else:
                result = self.graph.query(
                    """
                    MATCH (u:User)-[:HAS_SKILL]->(s:Skill)
                    WHERE s.name IN $skills AND u.id <> $exclude
                    OPTIONAL MATCH (u)-[:LIVES_IN]->(c:City)
                    RETURN u.id, u.alias, u.bio, collect(DISTINCT s.name) AS skills, c.name AS city
                    ORDER BY size(collect(DISTINCT s.name)) DESC
                    LIMIT $limit
                    """,
                    {
                        "skills": [s.lower().strip() for s in skills],
                        "exclude": exclude_user,
                        "limit": limit,
                    },
                )

            providers = []
            for row in result.result_set:
                providers.append({
                    "userId": row[0] or "",
                    "alias": row[1] or row[0] or "",
                    "bio": row[2] or "",
                    "skills": row[3] if isinstance(row[3], list) else [],
                    "city": row[4] or "",
                })
            return providers
        except Exception as e:
            print(f"[graph_store] search_providers error: {e}")
            return []
