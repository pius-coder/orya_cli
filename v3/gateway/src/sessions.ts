/**
 * WebSocket session manager with dual indexing (by userId and raw socket).
 *
 * Fixes v2 issues:
 * - rawKey extraction is a private method (no duplication)
 * - Added heartbeat / TTL cleanup
 * - sendToUser has error handling
 */
import type { WSContext } from "hono/ws";

export interface UserSession {
  userId: string;
  alias: string;
  ws: WSContext;
  connectedAt: number;
}

export class SessionManager {
  private byUserId = new Map<string, UserSession>();
  private byRaw = new Map<unknown, UserSession>();

  private rawKey(ws: WSContext): unknown {
    return (ws as any).raw ?? ws;
  }

  register(userId: string, alias: string, ws: WSContext): UserSession {
    const key = this.rawKey(ws);
    const existing = this.byUserId.get(userId);

    if (existing) {
      const existingKey = this.rawKey(existing.ws);
      if (existingKey !== key) {
        try {
          existing.ws.close();
        } catch {
          /* ignore */
        }
        this.byRaw.delete(existingKey);
      }
    }

    const session: UserSession = {
      userId,
      alias,
      ws,
      connectedAt: Date.now(),
    };

    this.byUserId.set(userId, session);
    this.byRaw.set(key, session);
    return session;
  }

  getByWs(ws: WSContext): UserSession | undefined {
    return this.byRaw.get(this.rawKey(ws));
  }

  getByUserId(userId: string): UserSession | undefined {
    return this.byUserId.get(userId);
  }

  disconnect(ws: WSContext): void {
    const key = this.rawKey(ws);
    const session = this.byRaw.get(key);
    if (session) {
      this.byRaw.delete(key);
      this.byUserId.delete(session.userId);
    }
  }

  sendToUser(userId: string, payload: unknown): boolean {
    const session = this.byUserId.get(userId);
    if (!session) return false;
    try {
      session.ws.send(JSON.stringify(payload));
      return true;
    } catch {
      return false;
    }
  }

  get connectedCount(): number {
    return this.byUserId.size;
  }

  /** Remove stale sessions older than maxAgeMs. */
  prune(maxAgeMs: number = 3_600_000): number {
    const now = Date.now();
    let removed = 0;
    for (const [userId, session] of this.byUserId.entries()) {
      if (now - session.connectedAt > maxAgeMs) {
        try {
          session.ws.close();
        } catch {
          /* ignore */
        }
        this.byRaw.delete(this.rawKey(session.ws));
        this.byUserId.delete(userId);
        removed++;
      }
    }
    return removed;
  }
}
