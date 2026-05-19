/**
 * Session Manager — tracks connected users and their WebSocket instances.
 *
 * Uses the raw Bun ServerWebSocket as key (WSContext.raw) for stable identity,
 * since Hono's WSContext wrapper may differ between callbacks.
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

  register(userId: string, alias: string, ws: WSContext): UserSession {
    const rawKey = (ws as any).raw ?? ws;

    // If same user reconnects, close old
    const existing = this.byUserId.get(userId);
    if (existing) {
      const existingRaw = (existing.ws as any).raw ?? existing.ws;
      if (existingRaw !== rawKey) {
        try { existing.ws.close(); } catch {}
        this.byRaw.delete(existingRaw);
      }
    }

    const session: UserSession = { userId, alias, ws, connectedAt: Date.now() };
    this.byUserId.set(userId, session);
    this.byRaw.set(rawKey, session);
    return session;
  }

  getByWs(ws: WSContext): UserSession | undefined {
    const rawKey = (ws as any).raw ?? ws;
    return this.byRaw.get(rawKey);
  }

  getByUserId(userId: string): UserSession | undefined {
    return this.byUserId.get(userId);
  }

  disconnect(ws: WSContext): void {
    const rawKey = (ws as any).raw ?? ws;
    const session = this.byRaw.get(rawKey);
    if (session) {
      this.byUserId.delete(session.userId);
      this.byRaw.delete(rawKey);
    }
  }

  sendToUser(userId: string, payload: any): boolean {
    const session = this.byUserId.get(userId);
    if (!session) return false;
    session.ws.send(JSON.stringify(payload));
    return true;
  }

  get connectedCount(): number {
    return this.byUserId.size;
  }
}
