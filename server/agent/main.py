"""Orya v2 — Agent FastAPI service.

Endpoints:
    POST /chat      — invoke the LangGraph for a user message
    POST /feedback  — record a good/bad rating
    GET  /health    — basic readiness probe

Lifespan:
    - Initialize Graphiti (FalkorDB)
    - Initialize PG asyncpg pool
    - Initialize LangGraph PostgresSaver checkpointer (with `setup()`)
    - Build & compile the graph
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from . import db
from .graph import build_graph_builder
from .models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    OptInDecision,
)
from .providers import build_llm, init_graphiti
from .settings import get_settings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("orya.agent")

# Module-level holder for components built during lifespan
_components: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings = get_settings()
    settings.configure_langsmith()

    logger.info("Boot: settings loaded.")

    # Postgres pool
    await db.init_pool()
    logger.info("Boot: PG pool ready.")

    # LangGraph checkpointer (uses raw psycopg URL)
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(
        settings.postgres_dsn_psycopg
    )
    checkpointer = await checkpointer_cm.__aenter__()
    try:
        await checkpointer.setup()
    except Exception:
        logger.exception("Checkpointer setup() failed — continuing best-effort.")
    _components["checkpointer_cm"] = checkpointer_cm
    _components["checkpointer"] = checkpointer
    logger.info("Boot: LangGraph PostgresSaver ready.")

    # Graphiti
    graphiti = await init_graphiti()
    _components["graphiti"] = graphiti
    logger.info("Boot: Graphiti ready.")

    # LLM (chat) and small LLM (intent classifier — same chain, smaller tokens)
    llm = build_llm()
    small_llm = build_llm(temperature=0.0, max_tokens=128)
    _components["llm"] = llm
    _components["small_llm"] = small_llm

    # Build & compile graph
    builder = build_graph_builder(
        graphiti=graphiti, llm_router=llm, small_llm=small_llm
    )
    graph = builder.compile(checkpointer=checkpointer)
    _components["graph"] = graph
    logger.info("Boot: graph compiled.")

    try:
        yield
    finally:
        logger.info("Shutdown: closing PG pool…")
        await db.close_pool()
        try:
            cm = _components.get("checkpointer_cm")
            if cm is not None:
                await cm.__aexit__(None, None, None)
        except Exception:
            logger.exception("Checkpointer close failed.")


app = FastAPI(title="Orya Agent", version="2.0.0", lifespan=lifespan)


# ============================================================
# Endpoints
# ============================================================


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    services = {
        "graphiti": _components.get("graphiti") is not None,
        "graph": _components.get("graph") is not None,
        "postgres": False,
    }
    try:
        pool = db.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        services["postgres"] = True
    except Exception:
        services["postgres"] = False
    return HealthResponse(ok=all(services.values()), services=services)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Track user
    try:
        await db.upsert_user(req.user_id, req.alias)
    except Exception:
        logger.exception("upsert_user failed (non-fatal).")

    # If this is an opt-in response, handle out-of-graph and short-circuit.
    if req.opt_in_response is not None:
        return await _handle_opt_in_response(req)

    graph = _components.get("graph")
    if graph is None:
        raise HTTPException(503, "graph not ready")

    config = {"configurable": {"thread_id": f"orya:{req.user_id}"}}
    initial: dict[str, Any] = {
        "messages": [HumanMessage(content=req.text)],
        "user_id": req.user_id,
        "user_alias": req.alias,
        "last_user_text": req.text,
        "last_assistant_reply": "",
        "facts_context": [],
        "extracted_facts": [],
        "intent": None,
        "candidates": [],
        "pending_opt_in": None,
        "opt_in_response": None,
        "trace": [],
    }

    final_state: dict[str, Any] = await graph.ainvoke(initial, config=config)

    return ChatResponse(
        reply=final_state.get("last_assistant_reply") or "",
        facts=[
            {
                "label": f["label"],
                "value": str(f["value"]),
                "confidence": float(f["confidence"]),
            }
            for f in final_state.get("extracted_facts") or []
        ],
        candidates=[
            {
                "user_id": c["user_id"],
                "alias": c.get("alias"),
                "summary": c.get("summary") or "",
                "score": float(c.get("score") or 0.0),
                "candidate_uuid": c.get("candidate_uuid") or "",
            }
            for c in final_state.get("candidates") or []
        ],
        pending_opt_in=final_state.get("pending_opt_in"),
        trace=[
            {"step": ev.get("step", ""), "detail": ev.get("detail")}
            for ev in final_state.get("trace") or []
        ],
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    try:
        await db.upsert_user(req.user_id)
        await db.record_feedback(
            user_id=req.user_id,
            user_text=req.user_text,
            assistant_reply=req.assistant_reply,
            rating=int(req.rating),
        )
    except Exception:
        logger.exception("record_feedback failed.")
        raise HTTPException(500, "failed to record feedback")
    return FeedbackResponse(ok=True)


# ============================================================
# Opt-in response handler (out-of-graph, simple state machine)
# ============================================================


async def _handle_opt_in_response(req: ChatRequest) -> ChatResponse:
    assert req.opt_in_response is not None
    opt_in_id = req.opt_in_response.opt_in_id
    decision = req.opt_in_response.decision

    row = await db.get_opt_in(opt_in_id)
    if row is None:
        raise HTTPException(404, "opt_in not found")

    if row["seeker_id"] == req.user_id:
        updated = await db.respond_seeker(opt_in_id, decision)
    elif row["provider_id"] == req.user_id:
        updated = await db.respond_provider(opt_in_id, decision)
    else:
        raise HTTPException(403, "user not part of this opt_in")

    if updated is None:
        raise HTTPException(409, "opt_in not in a state that accepts that decision")

    decision_text = (
        "ok j'ai noté ton choix"
        if decision == "accept"
        else "ok pas de souci, on note"
    )

    # Notify the counterpart
    await _notify_opt_in_counterpart(updated, originator=req.user_id)

    return ChatResponse(
        reply=decision_text,
        facts=[],
        candidates=[],
        pending_opt_in={
            "opt_in_id": str(updated["opt_in_id"]),
            "status": updated["status"],
        },
        trace=[{"step": "opt_in_response", "detail": updated["status"]}],
    )


async def _notify_opt_in_counterpart(
    row: dict[str, Any], originator: str
) -> None:
    s = get_settings()
    target_user = (
        row["provider_id"] if row["seeker_id"] == originator else row["seeker_id"]
    )
    payload = _make_opt_in_payload(row, target_user)
    if payload is None:
        return
    base = s.GATEWAY_INTERNAL_URL.rstrip("/")
    url = f"{base}/internal/push/{target_user}"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json=payload)
    except Exception:
        logger.exception("opt_in counterpart notify failed.")


def _make_opt_in_payload(
    row: dict[str, Any], target_user: str
) -> dict[str, Any] | None:
    status = row["status"]
    if status == "pending_provider":
        return {
            "type": "system",
            "text": (
                f"Quelqu'un cherche : '{row['need_summary']}'. Tu veux qu'on "
                f"te mette en relation ? (réponds 'oui' ou 'non')"
            ),
        }
    if status == "matched":
        return {
            "type": "system",
            "text": "C'est bon, vous êtes en relation. Bonne discussion !",
        }
    if status == "rejected_seeker":
        return None  # nothing to tell the provider
    if status == "rejected_provider":
        return {
            "type": "system",
            "text": "La personne contactée n'est pas dispo, je continue à chercher.",
        }
    return None


# Optional debug endpoint
@app.get("/debug/opt_ins/{user_id}")
async def list_opt_ins(user_id: str) -> list[dict[str, Any]]:
    return await db.list_pending_opt_ins(user_id)
