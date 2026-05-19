/**
 * Session Manager — tracks connected users and their WebSocket instances.
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
  private byWs = new Map<WSContext, UserSession>();

  register(userId: string, alias: string, ws: WSContext): UserSession {
    // If same user reconnects, close old
    const existing = this.byUserId.get(userId);
    if (existing && existing.ws !== ws) {
      try { existing.ws.close(); } catch {}
      this.byWs.delete(existing.ws);
    }

    const session: UserSession = { userId, alias, ws, connectedAt: Date.now() };
    this.byUserId.set(userId, session);
    this.byWs.set(ws, session);
    return session;
  }

  getByWs(ws: WSContext): UserSession | undefined {
    return this.byWs.get(ws);
  }

  getByUserId(userId: string): UserSession | undefined {
    return this.byUserId.get(userId);
  }

  disconnect(ws: WSContext): void {
    const session = this.byWs.get(ws);
    if (session) {
      this.byUserId.delete(session.userId);
      this.byWs.delete(ws);
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
