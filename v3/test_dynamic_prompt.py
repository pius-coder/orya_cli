"""Interactive prompt tester for Orya v3.

Run locally to test the persona prompt pipeline without the full server.
"""
import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

# Ensure agent is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.providers import build_llm
from agent.persona import build_messages


async def simulate_conversation() -> None:
    print("=== Orya v3 Prompt Tester ===")
    alias = input("Ton prénom: ").strip() or "toi"
    tutoyer_input = input("Tutoiement ? (o/n): ").strip().lower()
    tutoyer = tutoyer_input in ("o", "oui", "y", "yes", "")

    llm = build_llm(temperature=0.3, max_tokens=512)
    history: list = []

    print("\nParle à Orya (Ctrl+C pour quitter):\n")
    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break
        if not user_text:
            continue

        messages = build_messages(
            history=history,
            facts_context=[],
            good_examples=[],
            tutoyer=tutoyer,
            user_alias=alias,
        )
        messages.append(HumanMessage(content=user_text))

        try:
            response = await llm.ainvoke(messages)
            reply = str(getattr(response, "content", "")).strip()
        except Exception as e:
            reply = f"[Erreur LLM: {e}]"

        print(f"Orya: {reply}\n")
        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=reply))


if __name__ == "__main__":
    asyncio.run(simulate_conversation())
