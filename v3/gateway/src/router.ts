/**
 * Hono gateway router with typed events and unified chat handling.
 *
 * Fixes v2 issues:
 * - message and opt_in_response are unified into handleChatEvent
 * - ChatResponse typing is correct
 * - fetch has timeout and explicit typing
 * - event validation via narrow union type
 */
import type { WSContext } from "hono/ws";
import { SessionManager } from "./sessions";

const AGENT_URL = process.env.AGENT_URL ?? "http://127.0.0.1:5001";
const INTERNAL_API_KEY = process.env.INTERNAL_API_KEY ?? "";

// ── Types ─────────────────────────────────────────────────────────
interface ChatResponse {
  reply?: string;
  candidates?: Array<{
    user_id: string;
    alias?: string;
    summary: string;
    score: number;
    candidate_uuid?: string;
  }>;
  pending_opt_in?: Record<string, unknown>;
  trace?: Array<{ step: string; detail: string }>;
}

type WsEvent =
  | { type: "register"; userId?: string; alias?: string }
  | { type: "message"; text: string }
  | { type: "opt_in_response"; optInId: string; decision: "accept" | "reject"; summary?: string }
  | { type: "feedback"; rating: "good" | "bad"; userInput: string; oryaResponse: string }
  | { type: "tutoyer" }
  | { type: "ping" };

// ── Helpers ───────────────────────────────────────────────────────
function isValidEvent(data: unknown): data is WsEvent {
  if (typeof data !== "object" || data === null) return false;
  const d = data as Record<string, unknown>;
  if (typeof d.type !== "string") return false;
  switch (d.type) {
    case "register":
      return true;
    case "message":
      return typeof d.text === "string";
    case "opt_in_response":
      return (
        typeof d.optInId === "string" &&
        (d.decision === "accept" || d.decision === "reject")
      );
    case "feedback":
      return (
        (d.rating === "good" || d.rating === "bad") &&
        typeof d.userInput === "string" &&
        typeof d.oryaResponse === "string"
      );
    case "tutoyer":
    case "ping":
      return true;
    default:
      return false;
  }
}

async function callAgent(body: Record<string, unknown>): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  try {
    const res = await fetch(`${AGENT_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`Agent HTTP ${res.status}`);
    }
    return (await res.json()) as ChatResponse;
  } finally {
    clearTimeout(timeout);
  }
}

async function postFeedback(body: Record<string, unknown>): Promise<void> {
  try {
    await fetch(`${AGENT_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(10_000),
    });
  } catch (e) {
    console.warn("Feedback post failed:", e);
  }
}

function sendChatPayload(ws: WSContext, data: ChatResponse): void {
  ws.send(JSON.stringify({ type: "typing", value: false }));

  if (data.reply) {
    ws.send(JSON.stringify({ type: "reply", text: data.reply }));
  }

  if (data.candidates && data.candidates.length > 0) {
    ws.send(JSON.stringify({ type: "candidates", candidates: data.candidates }));
    if (data.pending_opt_in) {
      ws.send(
        JSON.stringify({ type: "pendingOptIn", payload: data.pending_opt_in })
      );
    }
  }

  if (data.trace) {
    for (const t of data.trace) {
      ws.send(JSON.stringify({ type: "trace", step: t.step, detail: t.detail }));
    }
  }
}

async function handleChatEvent(
  ws: WSContext,
  sessions: SessionManager,
  text: string,
  optInResponse?: { opt_in_id: string; decision: string }
): Promise<void> {
  const session = sessions.getByWs(ws);
  if (!session) {
    ws.send(JSON.stringify({ type: "system", level: "error", text: "Not registered" }));
    return;
  }

  ws.send(JSON.stringify({ type: "typing", value: true }));

  try {
    const payload: Record<string, unknown> = {
      user_id: session.userId,
      alias: session.alias,
      text,
    };
    if (optInResponse) {
      payload.opt_in_response = optInResponse;
    }
    const data = await callAgent(payload);
    sendChatPayload(ws, data);
  } catch (e) {
    console.error("Agent error:", e);
    ws.send(JSON.stringify({ type: "typing", value: false }));
    ws.send(
      JSON.stringify({
        type: "system",
        level: "error",
        text: "Agent unreachable. Retry in a moment.",
      })
    );
  }
}

// ── Main router ───────────────────────────────────────────────────
export function routeMessage(
  event: unknown,
  ws: WSContext,
  sessions: SessionManager
): void {
  if (!isValidEvent(event)) {
    ws.send(
      JSON.stringify({ type: "system", level: "warning", text: "Unknown event" })
    );
    return;
  }

  switch (event.type) {
    case "register": {
      const userId = event.userId ?? `anon-${Date.now()}`;
      const alias = event.alias ?? userId;
      sessions.register(userId, alias, ws);
      ws.send(
        JSON.stringify({
          type: "registered",
          userId,
          alias,
          addressForm: "tu",
        })
      );
      break;
    }

    case "message": {
      void handleChatEvent(ws, sessions, event.text);
      break;
    }

    case "opt_in_response": {
      void handleChatEvent(ws, sessions, event.summary ?? "(opt-in response)", {
        opt_in_id: event.optInId,
        decision: event.decision,
      });
      break;
    }

    case "feedback": {
      const session = sessions.getByWs(ws);
      if (!session) return;
      void postFeedback({
        user_id: session.userId,
        user_input: event.userInput,
        orya_response: event.oryaResponse,
        rating: event.rating,
      });
      break;
    }

    case "tutoyer": {
      ws.send(
        JSON.stringify({ type: "address_form", from: "tu", to: "tu" })
      );
      break;
    }

    case "ping": {
      ws.send(JSON.stringify({ type: "pong" }));
      break;
    }
  }
}
