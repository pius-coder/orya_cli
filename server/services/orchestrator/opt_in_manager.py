"""
Double Opt-In Manager — Gère le consentement des deux parties avant mise en relation.

Flow :
1. Orchestrator trouve un candidat pour un seeker
2. On demande au SEEKER : "j'ai quelqu'un, tu veux que je le contacte ?"
3. Si oui → on demande au PROVIDER : "hey, quelqu'un a besoin de toi, ça te dit ?"
4. Si les deux acceptent → on déclenche le tunnel (échange de contacts)
5. Si l'un refuse → on notifie l'autre gentiment

États d'un match :
- pending_seeker : en attente de la réponse du demandeur
- pending_provider : en attente de la réponse du prestataire
- both_accepted : les deux ont accepté
- declined : l'un a refusé
- expired : timeout (1h)
"""

import time
from typing import Optional


class OptInManager:
    def __init__(self):
        # matchId -> match state
        self.matches: dict[str, dict] = {}

    def create_match(
        self,
        match_id: str,
        seeker_id: str,
        seeker_alias: str,
        provider_id: str,
        provider_alias: str,
        reason: str = "",
    ) -> dict:
        """Create a new pending match."""
        match = {
            "id": match_id,
            "seeker_id": seeker_id,
            "seeker_alias": seeker_alias,
            "provider_id": provider_id,
            "provider_alias": provider_alias,
            "reason": reason,
            "status": "pending_seeker",
            "seeker_accepted": None,
            "provider_accepted": None,
            "created_at": time.time(),
        }
        self.matches[match_id] = match
        return match

    def respond(self, user_id: str, match_id: str, accept: bool) -> str:
        """
        User responds to opt-in.
        Returns: "waiting", "both_accepted", "declined"
        """
        match = self.matches.get(match_id)
        if not match:
            return "not_found"

        if user_id == match["seeker_id"]:
            match["seeker_accepted"] = accept
            if not accept:
                match["status"] = "declined"
                return "declined"
            # Seeker accepted → now ask provider
            match["status"] = "pending_provider"
            return "waiting"

        elif user_id == match["provider_id"]:
            match["provider_accepted"] = accept
            if not accept:
                match["status"] = "declined"
                return "declined"
            # Provider accepted → check if seeker also accepted
            if match["seeker_accepted"]:
                match["status"] = "both_accepted"
                return "both_accepted"
            match["status"] = "pending_seeker"
            return "waiting"

        return "not_found"

    def get_match(self, match_id: str) -> Optional[dict]:
        return self.matches.get(match_id)

    def get_pending_for_user(self, user_id: str) -> list[dict]:
        """Get all pending matches involving this user."""
        results = []
        for match in self.matches.values():
            if match["status"].startswith("pending"):
                if user_id in (match["seeker_id"], match["provider_id"]):
                    results.append(match)
        return results

    def cleanup_expired(self, max_age: float = 3600.0):
        """Remove matches older than max_age seconds."""
        now = time.time()
        expired = [
            mid for mid, m in self.matches.items()
            if now - m["created_at"] > max_age and m["status"].startswith("pending")
        ]
        for mid in expired:
            self.matches[mid]["status"] = "expired"
