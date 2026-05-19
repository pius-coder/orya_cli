/**
 * Message Router — dispatches CLI events to the appropriate backend service.
 *
 * Flow:
 *   CLI message → Gateway → Agent Orya (conversation response)
 *                         → Orchestrator (async extraction + background tasks)
 */

import type { WSContext } from "hono/ws";
import type { SessionManager } from "./sessions";

const AGENT_ORYA_URL = process.env.AGENT_ORYA_URL || "http://localhost:5001";
const ORCHESTRATOR_URL = process.env.ORCHESTRATOR_URL || "http://localhost:5002";

export async function routeMessage(event: any, ws: WSContext, sessions: SessionManager) {
  switch (event.type) {
    case "register": {
      const userId = event.userId || `anon-${Date.now()}`;
      const alias = event.alias || userId;
      sessions.register(userId, alias, ws);
      ws.send(JSON.stringify({
        type: "registered",
        userId,
        alias,
        addressForm: "tu", // Orya tutoie toujours — style humain
      }));
      console.log(`[router] registered ${userId}`);
      break;
    }

    case "message": {
      const session = sessions.getByWs(ws);
      if (!session) {
        ws.send(JSON.stringify({ type: "system", level: "error", text: "Not registered" }));
        return;
      }

      // Send "typing" indicator immediately
      ws.send(JSON.stringify({ type: "typing", value: true }));

      // 1. Call Agent Orya for conversational reply (sync)
      try {
        const resp = await fetch(`${AGENT_ORYA_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId: session.userId,
            text: event.text,
          }),
        });
        const data = await resp.json() as any;

        ws.send(JSON.stringify({ type: "typing", value: false }));
        ws.send(JSON.stringify({ type: "reply", text: data.reply }));

        // If Orya extracted facts inline, notify
        if (data.facts?.length) {
          for (const fact of data.facts) {
            ws.send(JSON.stringify({ type: "fact_recorded", fact }));
          }
        }
      } catch (err) {
        ws.send(JSON.stringify({ type: "typing", value: false }));
        ws.send(JSON.stringify({
          type: "system",
          level: "error",
          text: "Agent Orya unreachable",
        }));
        console.error("[router] agent-orya error:", err);
      }

      // 2. Fire & forget to Orchestrator (async extraction pipeline)
      fireAndForget(`${ORCHESTRATOR_URL}/process`, {
        userId: session.userId,
        text: event.text,
      });
      break;
    }

    case "tutoyer": {
      // Already in tu mode, acknowledge
      ws.send(JSON.stringify({ type: "address_form", from: "tu", to: "tu" }));
      break;
    }

    case "ping": {
      ws.send(JSON.stringify({ type: "pong" }));
      break;
    }

    default:
      ws.send(JSON.stringify({ type: "system", level: "warn", text: `Unknown event: ${event.type}` }));
  }
}

function fireAndForget(url: string, body: any) {
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch((err) => {
    console.warn(`[router] fire-and-forget failed: ${url}`, err.message);
  });
}
