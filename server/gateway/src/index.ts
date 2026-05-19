/**
 * Orya Gateway — Hono + Bun WebSocket
 *
 * Responsibilities:
 * 1. Accept CLI WebSocket connections
 * 2. Route messages to Agent Orya (conversation)
 * 3. Forward async events from Orchestrator back to the user
 * 4. Health check endpoint
 */

import { Hono } from "hono";
import { createBunWebSocket } from "hono/bun";
import type { ServerWebSocket } from "bun";
import { SessionManager } from "./sessions";
import { routeMessage } from "./router";

const { upgradeWebSocket, websocket } = createBunWebSocket<ServerWebSocket>();

const app = new Hono();
const sessions = new SessionManager();

// ── Health ────────────────────────────────────────────────────────
app.get("/health", (c) => c.json({ status: "ok", service: "gateway", ts: Date.now() }));

// ── WebSocket endpoint ────────────────────────────────────────────
app.get(
  "/ws",
  upgradeWebSocket((c) => ({
    onOpen(evt, ws) {
      console.log("[gateway] new connection");
    },
    onMessage(evt, ws) {
      const raw = typeof evt.data === "string" ? evt.data : new TextDecoder().decode(evt.data as ArrayBuffer);
      let parsed: any;
      try {
        parsed = JSON.parse(raw);
      } catch {
        ws.send(JSON.stringify({ type: "system", level: "error", text: "Invalid JSON" }));
        return;
      }
      routeMessage(parsed, ws, sessions);
    },
    onClose(evt, ws) {
      sessions.disconnect(ws);
      console.log("[gateway] connection closed");
    },
    onError(evt, ws) {
      console.error("[gateway] ws error", evt);
    },
  }))
);

// ── Internal API (called by services to push events to users) ─────
app.post("/internal/push/:userId", async (c) => {
  const userId = c.req.param("userId");
  const body = await c.req.json();
  const sent = sessions.sendToUser(userId, body);
  if (!sent) return c.json({ error: "user not connected" }, 404);
  return c.json({ ok: true });
});

// ── Start ─────────────────────────────────────────────────────────
const port = parseInt(process.env.GATEWAY_PORT || "4001");

export default {
  port,
  fetch: app.fetch,
  websocket,
};

console.log(`[gateway] listening on :${port}`);
