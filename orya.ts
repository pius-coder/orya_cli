/**
 * Sandbox CLI client.
 *
 * Connects to the sandbox server over WebSocket and lets you chat with
 * Orya from the terminal. Mirrors a WhatsApp conversation: typing
 * indicators, candidate cards, system traces.
 *
 *   bun orya.ts                     (default user)
 *   bun orya.ts --user user-cli2   (named user)
 *   bun orya.ts --trace             (verbose trace events)
 *
 * Type messages and press Enter to send.
 *   /quit       → exit
 *   /trace      → toggle trace events
 *   /pick N     → pick candidate N from the last cards
 *   /tutoyer    → accept tutoiement
 *   /vouvoyer   → refuse tutoiement
 *   /facts      → ask the server to dump known facts (synthetic message)
 */



import readline from "node:readline";
import process from "node:process";
import type { ClientEvent, ServerEvent, Candidate } from "./types.ts";

// ── Args ──────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
let userId = "user-cli";
let trace = false;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--user" && argv[i + 1]) {
    userId = argv[i + 1];
    i++;
  } else if (argv[i] === "--trace") {
    trace = true;
  } else if (argv[i] === "--help" || argv[i] === "-h") {
    console.log("Usage: bun orya.ts [--user <id>] [--trace]");
    process.exit(0);
  }
}

const SERVER = process.env.SANDBOX_WS ?? `ws://${process.env.SANDBOX_HOST ?? "localhost"}:${process.env.SANDBOX_PORT ?? "4001"}/ws`;

// ── Colors ────────────────────────────────────────────────────────
const C = {
  dim:   (s: string) => `\x1b[2m${s}\x1b[0m`,
  bold:  (s: string) => `\x1b[1m${s}\x1b[0m`,
  cyan:  (s: string) => `\x1b[36m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  yel:   (s: string) => `\x1b[33m${s}\x1b[0m`,
  red:   (s: string) => `\x1b[31m${s}\x1b[0m`,
  mag:   (s: string) => `\x1b[35m${s}\x1b[0m`,
  blue:  (s: string) => `\x1b[34m${s}\x1b[0m`,
};

// ── Connection ────────────────────────────────────────────────────
let lastCandidates: Candidate[] = [];
let lastSessionId = "";
let lastPendingOptInId: string | null = null;
/** Timestamp of the last user message sent. Set when the CLI sends a
 * "message" event, cleared when the first reply arrives or turn.done. */
let turnStartedAt: number | null = null;
let firstReplyShown = false;
let thinkingInterval: ReturnType<typeof setInterval> | null = null;

function startThinking() {
  if (thinkingInterval) return;
  thinkingInterval = setInterval(() => {
    if (!turnStartedAt) return;
    const ms = Date.now() - turnStartedAt;
    const display = ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
    process.stdout.write(`\r\x1b[K${C.dim(`Orya réfléchit… ${display}`)}`);
  }, 250);
}

function stopThinking() {
  if (thinkingInterval) {
    clearInterval(thinkingInterval);
    thinkingInterval = null;
  }
  process.stdout.write("\r\x1b[K");
}

console.log(C.dim(`→ connecting to ${SERVER}…`));
const ws = new WebSocket(SERVER);

ws.addEventListener("open", () => {
  console.log(C.green("✓ connected"));
  const reg: ClientEvent = { type: "register", userId };
  ws.send(JSON.stringify(reg));
});

ws.addEventListener("error", (e) => {
  console.error(C.red("✗ websocket error"), e);
});

ws.addEventListener("close", () => {
  console.log(C.dim("\n— connection closed —"));
  process.exit(0);
});

ws.addEventListener("message", (ev) => {
  let msg: ServerEvent;
  try {
    msg = JSON.parse(typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data as ArrayBuffer));
  } catch {
    return;
  }
  render(msg);
});

// ── Rendering ─────────────────────────────────────────────────────
function clearLine() {
  process.stdout.write("\r\x1b[K");
}

function prompt(): void {
  rl.prompt(true);
}

function render(ev: ServerEvent): void {
  clearLine();
  switch (ev.type) {
    case "registered":
      console.log(C.cyan(`◆ Connecté en tant que ${ev.alias} (${ev.userId})  [forme: ${ev.addressForm}]`));
      console.log(C.dim("Tape /help pour les commandes."));
      break;

    case "typing":
      if (ev.value) process.stdout.write(C.dim("Orya écrit…"));
      break;

    case "reply": {
      stopThinking();
      let suffix = "";
      if (!firstReplyShown && turnStartedAt) {
        const ms = Date.now() - turnStartedAt;
        const display = ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
        suffix = "  " + C.dim(`(${display})`);
        firstReplyShown = true;
      }
      console.log(C.bold(C.green("Orya  ")) + ev.text + suffix);
      break;
    }

    case "candidates": {
      const candidates = ev.items ?? (ev as any).candidates ?? [];
      lastCandidates = candidates;
      lastSessionId = (ev as any).sessionId ?? "";
      if (!candidates.length) {
        console.log(C.dim("[système] Recherche de candidats lancée mais aucun résultat."));
        break;
      }
      console.log(C.bold(C.mag(`📇 ${candidates.length} prestataire${candidates.length > 1 ? "s" : ""} (tour ${(ev as any).tour ?? 1})`)));
      candidates.forEach((c: any, i: number) => {
        const score = ((c.score ?? c.scoreFused ?? 0) * 100).toFixed(0);
        console.log(C.mag(` ${i + 1}. `) + C.bold(c.alias ?? c.user_id ?? "?") + C.dim(`  score=${score}%`));
        console.log(C.dim(`    ${c.summary ?? c.bio ?? ""}`));
        if (c.skills) console.log(C.dim(`    skills: ${(c.skills ?? []).join(", ")}  ·  ville: ${c.city ?? "?"}`));
      });
      console.log(C.dim("Réponds avec le numéro (ou /pick N) pour choisir."));
      break;
    }

    case "system": {
      console.log(
        (ev.level === "error" ? C.red : ev.level === "warn" ? C.yel : C.dim)(`[${ev.level}] ${ev.text}`),
      );
      if ((ev as any).opt_in_id) {
        lastPendingOptInId = (ev as any).opt_in_id;
      }
      break;
    }

    case "trace":
      if (ev.node === "turn.done") {
        stopThinking();
        turnStartedAt = null;
        firstReplyShown = false;
        if (trace) {
          console.log(C.dim(`· turn.done ${JSON.stringify(ev.payload)}`));
        }
      } else if (trace) {
        console.log(C.dim(`· ${ev.node} ${JSON.stringify(ev.payload)}`));
      }
      break;

    case "fact_recorded":
      if (trace) {
        console.log(
          C.blue(`✎ fact ${ev.fact.kind}=${ev.fact.value}  (${(ev.fact.confidence * 100).toFixed(0)}%)`),
        );
      }
      break;

    case "address_form":
      console.log(C.dim(`◇ forme d'adresse: ${ev.from} → ${ev.to}`));
      break;

    case "pong":
      // ignore
      break;
  }
  prompt();
}

// ── REPL ──────────────────────────────────────────────────────────
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: C.cyan("Vous » "),
});

rl.on("line", (raw) => {
  const text = raw.trim();
  if (!text) {
    prompt();
    return;
  }

  if (text.startsWith("/")) {
    handleCommand(text);
    return;
  }

  // Intercept yes/no answers for pending opt-in requests
  if (lastPendingOptInId) {
    const lower = text.toLowerCase();
    if (lower === "oui" || lower === "non" || lower === "y" || lower === "n" || lower === "o") {
      const decision = (lower === "oui" || lower === "y" || lower === "o") ? "accept" : "reject";
      const ev = {
        type: "opt_in_response",
        optInId: lastPendingOptInId,
        decision: decision,
        summary: text,
      };
      lastPendingOptInId = null; // consume
      turnStartedAt = Date.now();
      firstReplyShown = false;
      startThinking();
      ws.send(JSON.stringify(ev));
      return;
    }
  }

  if (ws.readyState !== WebSocket.OPEN) {
    console.log(C.red("Pas connecté. Réessaie dans un instant."));
    prompt();
    return;
  }

  const ev: ClientEvent = { type: "message", text };
  turnStartedAt = Date.now();
  firstReplyShown = false;
  startThinking();
  ws.send(JSON.stringify(ev));
});

rl.on("close", () => {
  process.exit(0);
});

function handleCommand(cmd: string) {
  const [head, ...rest] = cmd.split(/\s+/);
  switch (head) {
    case "/quit":
    case "/exit":
      ws.close();
      rl.close();
      return;
    case "/help":
      console.log(
        [
          "Commandes :",
          "  /quit        Quitter",
          "  /trace       Activer/désactiver le mode trace",
          "  /pick N      Choisir le candidat N",
          "  /tutoyer     Accepter le tutoiement",
          "  /vouvoyer    Refuser le tutoiement",
          "  /facts       Demander à Orya ce qu'il sait sur toi",
        ].join("\n"),
      );
      prompt();
      return;
    case "/trace":
      trace = !trace;
      console.log(C.dim(`trace = ${trace}`));
      prompt();
      return;
    case "/pick": {
      const n = Number(rest[0]);
      if (!n || n < 1 || n > lastCandidates.length) {
        console.log(C.red("Numéro invalide. Redemande des cartes d'abord."));
        prompt();
        return;
      }
      const candidate = lastCandidates[n - 1] as any;
      const optInId = candidate.opt_in_id ?? lastSessionId;
      if (!optInId) {
        console.log(C.red("Pas d'ID de mise en relation trouvé pour ce candidat."));
        prompt();
        return;
      }
      const ev = {
        type: "opt_in_response",
        optInId: optInId,
        decision: "accept",
        summary: `Choix du prestataire ${candidate.alias || candidate.user_id}`
      };
      turnStartedAt = Date.now();
      firstReplyShown = false;
      startThinking();
      ws.send(JSON.stringify(ev));
      console.log(C.dim(`-> Validation de la proposition de mise en relation (Opt-in ID: ${optInId})...`));
      return;
    }
    case "/tutoyer": {
      const ev: ClientEvent = { type: "tutoyer", accept: true };
      turnStartedAt = Date.now();
      firstReplyShown = false;
      startThinking();
      ws.send(JSON.stringify(ev));
      return;
    }
    case "/vouvoyer": {
      const ev: ClientEvent = { type: "tutoyer", accept: false };
      turnStartedAt = Date.now();
      firstReplyShown = false;
      startThinking();
      ws.send(JSON.stringify(ev));
      return;
    }
    case "/facts": {
      const ev: ClientEvent = {
        type: "message",
        text: "Au fait, qu'est-ce que vous savez de moi pour le moment ?",
      };
      turnStartedAt = Date.now();
      firstReplyShown = false;
      startThinking();
      ws.send(JSON.stringify(ev));
      return;
    }
    default:
      console.log(C.red(`Commande inconnue : ${head}`));
      prompt();
  }
}

// Suppress unused warning: lastSessionId might be useful later.
void lastSessionId;
