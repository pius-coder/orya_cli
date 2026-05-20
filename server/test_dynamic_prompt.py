import asyncio
import os
import sys
from dotenv import load_dotenv

# Désactiver le tracing LangChain pour éviter les erreurs de logs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Ajouter le chemin de l'agent
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

load_dotenv()

from agent.providers import build_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.persona import build_messages

# Modèles LLM configurés
llm = build_llm(temperature=0.3)  # Température abaissée pour moins de créativité/blabla


# 4. Simulation de la conversation interactive
async def simulate_conversation():
    print("=" * 70)
    print(" CHAT INTERACTIF SIMPLIFIÉ (PROMPT IMPLICITE) - ORYA")
    print("=" * 70)
    print("Tapez 'quitter' ou 'exit' pour arrêter.\n")
    
    # Configuration par l'utilisateur
    alias_input = input("Entrez votre prénom / alias (par défaut: Dominique) : ").strip()
    user_alias = alias_input if alias_input else "Dominique"
    
    tutoyer_input = input("Voulez-vous tutoyer Orya ? (O/N, par défaut: N) : ").strip().upper()
    tutoyer = (tutoyer_input == "O")
    
    facts = [
        f"{user_alias} est un entrepreneur.",
        f"{user_alias} s'intéresse aux opportunités de son réseau."
    ]
    
    print(f"\n[Configuration] Mode : {'TUTOIEMENT' if tutoyer else 'VOUVOIEMENT'}")
    print(f"[Configuration] Alias : {user_alias}")
    print(f"[Configuration] Faits : {facts}\n")
    print("Orya est prête. Discutez avec elle !")
    print("=" * 70)
    
    history_messages = []
    
    while True:
        try:
            msg = input("\nVous » ").strip()
            if not msg:
                continue
            if msg.lower() in ["quitter", "exit", "quit"]:
                print("Fin de la session de chat interactif.")
                break
            
            # Appliquer le même pipeline de messages que persona_respond.py
            # en injectant l'historique et les faits
            history = list(history_messages)
            messages = build_messages(
                history=history,
                facts_context=facts,
                tutoyer=tutoyer,
                user_alias=user_alias,
            )
            messages.append(HumanMessage(content=msg))
            
            # D. Générer la réponse
            ai_resp = await llm.ainvoke(messages)
            reply = ai_resp.content.strip()
            
            print(f"Orya » {reply}")
            
            # Mettre à jour l'historique
            history_messages.append(HumanMessage(content=msg))
            history_messages.append(AIMessage(content=reply))
            
        except KeyboardInterrupt:
            print("\nSession interrompue. Fin du chat.")
            break
        except Exception as e:
            print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    asyncio.run(simulate_conversation())
