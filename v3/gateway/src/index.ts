/**
 * Hono + Bun WebSocket gateway for Orya v3.
 *
 * Fixes v2 issues:
 * - Explicit Bun.serve() call (no implicit auto-start magic)
 * - Basic auth on /internal/push via INTERNAL_API_KEY
 * - Prunes stale sessions every 5 minutes
 * - Typed WS events
 */
import { Hono } from "hono";
import { createBunWebSocket } from "hono/bun";
import type { ServerWebSocket } from "bun";
import { routeMessage } from "./router";
import { SessionManager } from "./sessions";

const { upgradeWebSocket, websocket } = createBunWebSocket<ServerWebSocket>();
const sessions = new SessionManager();
const app = new Hono();
const INTERNAL_API_KEY = process.env.INTERNAL_API_KEY ?? "";

// ── Health ────────────────────────────────────────────────────────
app.get("/health", (c) => {
  return c.json({
    status: "ok",
    service: "orya-gateway",
    ts: new Date().toISOString(),
    connections: sessions.connectedCount,
  });
});

// ── WebSocket ─────────────────────────────────────────────────────
app.get(
  "/ws",
  upgradeWebSocket((c) => ({
    onOpen(_evt, ws) {
      console.log("WS connection opened");
    },
    onMessage(evt, ws) {
      let payload: unknown;
      try {
        const text =
          typeof evt.data === "string"
            ? evt.data
            : new TextDecoder().decode(evt.data);
        payload = JSON.parse(text);
      } catch {
        ws.send(
          JSON.stringify({
            type: "system",
            level: "error",
            text: "Invalid JSON",
          })
        );
        return;
      }
      routeMessage(payload, ws, sessions);
    },
    onClose(_evt, ws) {
      sessions.disconnect(ws);
      console.log("WS connection closed");
    },
    onError(_evt, ws) {
      console.error("WS error");
    },
  }))
);

// ── Internal push ─────────────────────────────────────────────────
app.post("/internal/push/:userId", async (c) => {
  if (INTERNAL_API_KEY) {
    const auth = c.req.header("x-internal-api-key");
    if (auth !== INTERNAL_API_KEY) {
      return c.json({ error: "Unauthorized" }, 401);
    }
  }

  const userId = c.req.param("userId");
  let body: unknown;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body" }, 400);
  }

  const ok = sessions.sendToUser(userId, body);
  if (!ok) {
    return c.json({ error: "User not connected" }, 404);
  }
  return c.json({ ok: true });
});

// ── Start server ──────────────────────────────────────────────────
const port = parseInt(process.env.GATEWAY_PORT ?? "4001", 10);

const server = Bun.serve({
  port,
  fetch: app.fetch,
  websocket,
});

console.log(`Orya Gateway listening on ws://localhost:${port}/ws`);

// Prune stale sessions every 5 minutes
setInterval(() => {
  const removed = sessions.prune(3_600_000); // 1 hour TTL
  if (removed > 0) {
    console.log(`Pruned ${removed} stale sessions`);
  }
}, 300_000);
