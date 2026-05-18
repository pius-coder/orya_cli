/**
 * Sandbox types — shared between server and CLI.
 */

export type Language = "FR" | "EN";

/**
 * How Orya addresses the user.
 *  - "vous"        : default polite form
 *  - "negotiating" : Orya has just asked permission to switch to "tu"
 *  - "tu"          : permission granted, casual form
 */
export type AddressForm = "vous" | "negotiating" | "tu";

export type SearchStatus =
  | "idle"
  | "searching"
  | "presenting"
  | "matched"
  | "no_results";

export interface UserProfile {
  id: string;
  alias: string;
  bio: string;
  isProvider: boolean;
  skills: string[];
  city: string;
}

export interface Candidate {
  userId: string;
  alias: string;
  bio: string;
  skills: string[];
  city: string;
  scoreGraph: number;
  scoreVector: number;
  scoreFused: number;
  rank: number;
}

export interface MatchConstraints {
  skills: string[];
  city: string | null;
  language: Language;
}

/** Anything Orya learns about the user that we want to keep. */
export type FactKind =
  | "skill"
  | "city"
  | "need"
  | "frustration"
  | "preference"
  | "personal"; // hobbies, family, work context…

export interface ExtractedFact {
  kind: FactKind;
  value: string;
  confidence: number;
  source: "inline" | "tool" | "manual";
  ts: number;
}

export interface MatchSession {
  id: string;
  userId: string;
  query: string;
  candidates: Candidate[];
  status: SearchStatus;
  tour: number;
  createdAt: number;
}

// ------------------------------------------------------------------
// CLI ↔ Server WebSocket protocol
// ------------------------------------------------------------------

export type ClientEvent =
  | { type: "register"; userId: string; alias?: string }
  | { type: "message"; text: string }
  | { type: "select"; sessionId: string; candidateUserId: string }
  | { type: "tutoyer"; accept: boolean } // user replies to Orya's tutoyer offer
  | { type: "ping" };

export type ServerEvent =
  | { type: "registered"; userId: string; alias: string; addressForm: AddressForm }
  | { type: "typing"; value: boolean }
  | { type: "reply"; text: string; meta?: Record<string, unknown> }
  | {
      type: "candidates";
      sessionId: string;
      candidates: Candidate[];
      tour: number;
      reason?: string;
    }
  | { type: "system"; level: "info" | "warn" | "error"; text: string }
  | { type: "trace"; node: string; payload: Record<string, unknown> }
  | { type: "fact_recorded"; fact: ExtractedFact }
  | { type: "address_form"; from: AddressForm; to: AddressForm }
  | { type: "pong" };
