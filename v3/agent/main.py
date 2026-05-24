"""FastAPI service for Orya v3 with MemBrain integration.

Updated to initialize embedder and pass it to the graph.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from . import db
from .core.config import get_settings
from .graph import build_graph
from .manifests.registry import ManifestRegistry
from .models import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse, HealthResponse
from .providers import build_llm, init_graphiti
from .providers.embedder import HuggingFaceEmbedder, HuggingFaceEmbedderConfig

logger = logging.getLogger(__name__)
_components: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.configure_langsmith()

    await db.init_pool()
    async with AsyncPostgresSaver.from_conn_string(settings.postgres_dsn_psycopg) as checkpointer:
        await checkpointer.setup()
        graphiti = await init_graphiti()
        llm = build_llm()

        # MemBrain embedder
        embedder = HuggingFaceEmbedder(
            config=HuggingFaceEmbedderConfig(
                api_key=settings.HUGGINGFACE_API_KEY,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dim=384,
            )
        )

        import pathlib

        manifests_dir = pathlib.Path(__file__).parent / "manifests"
        manifests = ManifestRegistry(str(manifests_dir))
        graph = build_graph(graphiti=graphiti, llm=llm, manifests=manifests, embedder=embedder)

        _components.update(
            {
                "settings": settings,
                "pool": db.get_pool(),
                "checkpointer": checkpointer,
                "graphiti": graphiti,
                "llm": llm,
                "embedder": embedder,
                "manifests": manifests,
                "graph": graph,
            }
        )

        logger.info("Orya agent v3+MemBrain started")
        yield

    await db.close_pool()
    logger.info("Orya agent v3+MemBrain shutdown")


app = FastAPI(title="Orya Agent v3+MemBrain", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    services: dict[str, bool] = {"agent": True}
    try:
        db.get_pool()
        services["postgres"] = True
    except Exception:
        services["postgres"] = False
    try:
        gt = _components.get("graphiti")
        services["graphiti"] = gt is not None
    except Exception:
        services["graphiti"] = False
    return HealthResponse(ok=all(services.values()), services=services)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    graph = _components["graph"]
    await db.upsert_user(req.user_id, req.alias)

    if req.opt_in_response:
        return await _handle_opt_in_response(req)

    state = {
        "messages": [HumanMessage(content=req.text)],
        "user_id": req.user_id,
        "user_alias": req.alias,
        "last_user_text": req.text,
        "last_assistant_reply": "",
        "strategy": "chat",
        "match_query": "",
        "user_reflection": None,
        "orya_reflection": None,
        "facts_context": None,
        "tool_calls": [],
        "candidates": [],
        "pending_opt_in": None,
        "opt_in_response": None,
        "trace": [],
    }

    try:
        final_state = await graph.ainvoke(
            state,
            config={"configurable": {"thread_id": req.user_id}},
        )
    except Exception as e:
        logger.error("Graph invocation failed: %s", e)
        raise HTTPException(status_code=500, detail="Agent processing failed") from e

    reply = final_state.get("last_assistant_reply", "")
    candidates = final_state.get("candidates", [])
    pending = final_state.get("pending_opt_in")
    trace = final_state.get("trace", [])

    from .models.schemas import CandidateOut, TraceEvent

    typed_candidates = [
        CandidateOut(
            user_id=c.get("user_id", ""),
            alias=c.get("alias"),
            summary=c.get("summary", ""),
            score=c.get("score", 0.0),
            candidate_uuid=c.get("candidate_uuid"),
        )
        for c in candidates
    ]
    typed_trace = [TraceEvent(step=t.get("step", ""), detail=t.get("detail", "")) for t in trace]

    return ChatResponse(
        reply=reply,
        candidates=typed_candidates,
        pending_opt_in=pending,
        trace=typed_trace,
    )


async def _handle_opt_in_response(req: ChatRequest) -> ChatResponse:
    if not req.opt_in_response:
        raise HTTPException(status_code=400, detail="Missing opt_in_response")

    opt_in = await db.get_opt_in(req.opt_in_response.opt_in_id)
    if not opt_in:
        return ChatResponse(reply="Je n'ai pas trouvé ce matching.")

    user_id = req.user_id
    decision = req.opt_in_response.decision == "accept"

    if opt_in["seeker_id"] == user_id and opt_in["status"] == "pending_seeker":
        updated = await db.respond_seeker(str(opt_in["id"]), decision)
        if updated and updated["status"] == "pending_provider":
            await _notify_counterpart(
                target_user_id=str(updated["provider_id"]),
                payload={
                    "type": "opt_in_request",
                    "opt_in_id": str(updated["id"]),
                    "summary": updated.get("reason", ""),
                },
            )
            return ChatResponse(reply="OK, je demande à l'autre personne.")
        return ChatResponse(reply="Pas de souci, je note.")

    elif opt_in["provider_id"] == user_id and opt_in["status"] == "pending_provider":
        updated = await db.respond_provider(str(opt_in["id"]), decision)
        if updated and updated["status"] == "both_accepted":
            await _notify_counterpart(
                target_user_id=str(updated["seeker_id"]),
                payload={"type": "match_confirmed", "opt_in_id": str(updated["id"])},
            )
            await _notify_counterpart(
                target_user_id=str(updated["provider_id"]),
                payload={"type": "match_confirmed", "opt_in_id": str(updated["id"])},
            )
            return ChatResponse(reply="Super, c'est bon pour vous deux ! Je vous mets en relation.")
        return ChatResponse(reply="D'accord, pas de souci.")

    return ChatResponse(reply="Ce matching n'est plus actif.")


async def _notify_counterpart(target_user_id: str, payload: dict[str, Any]) -> None:
    settings = get_settings()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.INTERNAL_API_KEY:
        headers["x-internal-api-key"] = settings.INTERNAL_API_KEY
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{settings.GATEWAY_INTERNAL_URL}/internal/push/{target_user_id}",
                json=payload,
                headers=headers,
            )
    except Exception as e:
        logger.error("Notify counterpart failed: %s", e)


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    await db.record_feedback(req.user_id, req.user_input, req.orya_response, req.rating)
    return FeedbackResponse(ok=True)
