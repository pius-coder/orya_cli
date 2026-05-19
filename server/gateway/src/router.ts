/**
 * Message Router — dispatches CLI events to the LangGraph agent.
 *
 * v2 simplification: a single sync call to /chat returns the persona reply
 * plus structured facts/candidates/trace. Async events (e.g. opt-in
 * counterpart notifications) reach the user via /internal/push.
 */

import type { WSContext } from "hono/ws";
import type { SessionManager } from "./sessions";

const AGENT_URL = process.env.AGENT_URL || "http://127.0.0.1:5001";

interface ChatResponse {
  reply: string;
  facts: Array<{ label: string; value: string; confidence: number }>;
  candidates: Array<{
    user_id: string;
    alias?: string | null;
    summary: string;
    score: number;
    candidate_uuid: string;
  }>;
  pending_opt_in?: Record<string, unknown> | null;
  trace?: Array<{ step: string; detail?: string | null }>;
}

export async function routeMessage(
  event: any,
  ws: WSContext,
  sessions: SessionManager,
) {
  switch (event.type) {
    case "register": {
      const userId = event.userId || `anon-${Date.now()}`;
      const alias = event.alias || userId;
      sessions.register(userId, alias, ws);
      ws.send(
        JSON.stringify({
          type: "registered",
          userId,
          alias,
          addressForm: "tu",
        }),
      );
      console.log(`[router] registered ${userId}`);
      break;
    }

    case "message": {
      const session = sessions.getByWs(ws);
      if (!session) {
        ws.send(
          JSON.stringify({
            type: "system",
            level: "error",
            text: "Not registered",
          }),
        );
        return;
      }
      ws.send(JSON.stringify({ type: "typing", value: true }));
      try {
        const data = await callAgent({
          user_id: session.userId,
          alias: session.alias,
          text: event.text,
          opt_in_response: event.optInResponse,
        });
        sendChatPayload(ws, data);
      } catch (err) {
        ws.send(JSON.stringify({ type: "typing", value: false }));
        ws.send(
          JSON.stringify({
            type: "system",
            level: "error",
            text: "Agent unreachable",
          }),
        );
        console.error("[router] agent error:", err);
      }
      break;
    }

    case "opt_in_response": {
      const session = sessions.getByWs(ws);
      if (!session) {
        ws.send(
          JSON.stringify({
            type: "system",
            level: "error",
            text: "Not registered",
          }),
        );
        return;
      }
      ws.send(JSON.stringify({ type: "typing", value: true }));
      try {
        const data = await callAgent({
          user_id: session.userId,
          alias: session.alias,
          text: event.summary || "(opt-in response)",
          opt_in_response: {
            opt_in_id: event.optInId,
            decision: event.decision,
          },
        });
        sendChatPayload(ws, data);
      } catch (err) {
        ws.send(JSON.stringify({ type: "typing", value: false }));
        ws.send(
          JSON.stringify({
            type: "system",
            level: "error",
            text: "Agent unreachable",
          }),
        );
        console.error("[router] agent error:", err);
      }
      break;
    }

    case "feedback": {
      const session = sessions.getByWs(ws);
      if (!session) return;
      void postFeedback({
        user_id: session.userId,
        user_text: event.userText,
        assistant_reply: event.assistantReply,
        rating: event.rating,
      });
      break;
    }

    case "tutoyer": {
      ws.send(JSON.stringify({ type: "address_form", from: "tu", to: "tu" }));
      break;
    }

    case "ping": {
      ws.send(JSON.stringify({ type: "pong" }));
      break;
    }

    default:
      ws.send(
        JSON.stringify({
          type: "system",
          level: "warn",
          text: `Unknown event: ${event.type}`,
        }),
      );
  }
}

async function callAgent(body: Record<string, unknown>): Promise<ChatResponse> {
  const resp = await fetch(`${AGENT_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    throw new Error(`agent ${resp.status}`);
  }
  return (await resp.json()) as ChatResponse;
}

function sendChatPayload(ws: WSContext, data: ChatResponse) {
  ws.send(JSON.stringify({ type: "typing", value: false }));
  if (data.reply) {
    ws.send(JSON.stringify({ type: "reply", text: data.reply }));
  }
  for (const fact of data.facts || []) {
    ws.send(
      JSON.stringify({
        type: "fact_recorded",
        label: fact.label,
        value: fact.value,
        confidence: fact.confidence,
      }),
    );
  }
  if (data.candidates && data.candidates.length > 0) {
    ws.send(
      JSON.stringify({
        type: "candidates",
        items: data.candidates,
        pendingOptIn: data.pending_opt_in ?? null,
      }),
    );
  }
  for (const ev of data.trace || []) {
    ws.send(
      JSON.stringify({
        type: "trace",
        step: ev.step,
        detail: ev.detail ?? undefined,
      }),
    );
  }
}

async function postFeedback(body: Record<string, unknown>) {
  try {
    await fetch(`${AGENT_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    console.warn("[router] feedback post failed:", err);
  }
}
