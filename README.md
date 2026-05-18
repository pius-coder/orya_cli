# Orya CLI

Terminal client to chat with **Orya** — a friendly conversational agent —
through a Cloudflare-tunneled sandbox server.

```
┌──────────────┐  WebSocket   ┌────────────────────────────────┐
│  orya CLI    │ ──────────▶  │  Cloudflare Tunnel             │
│  (you)       │ ◀──────────  │  ↳ private Orya sandbox server │
└──────────────┘   wss://     └────────────────────────────────┘
```

## Prerequisites

- [Bun](https://bun.sh) ≥ 1.1 (`curl -fsSL https://bun.sh/install | bash`)
- The WebSocket URL of the server (your friend who's hosting will give it
  to you, e.g. `wss://orya-cli.globalimex.online/ws`)

## Install

```bash
git clone https://github.com/pius-coder/orya_cli.git
cd orya_cli
bun install
```

## Run

Point at the server endpoint and launch the chat:

```bash
export SANDBOX_WS=wss://orya-cli.globalimex.online/ws
bun start
```

Or pass a named user:

```bash
SANDBOX_WS=wss://orya-cli.globalimex.online/ws bun orya.ts --user friend-jean
```

Add `--trace` to see internal events (extracted facts, LLM provider/model,
tool calls).

## Commands

Inside the chat:

| Command | Effect |
|---------|--------|
| `/quit` | Exit |
| `/trace` | Toggle verbose trace events |
| `/pick N` | Pick candidate N from the last cards |
| `/tutoyer` | Accept tutoiement when Orya offers it |
| `/vouvoyer` | Decline tutoiement |
| `/facts` | Ask Orya what they remember about you |

Just type and press Enter to send a message.

## What to expect

- Orya starts in **vous** form and may propose to switch to **tu** after a
  few warm exchanges.
- Every message is analysed in real-time: skills, city, frustrations,
  preferences, personal mentions are remembered.
- If you ask for a service provider (plumber, electrician, mechanic…),
  Orya will surface candidate cards. Reply with the card number to
  confirm.

## Troubleshooting

- **`websocket error`** → the server is down or the URL is wrong. Ask
  the host.
- **Long first reply** → the embedding model cold-starts (~3-5 s) on the
  first turn. Subsequent turns are fast.
- **No `bun` command** → install Bun (link above).

## License

MIT
