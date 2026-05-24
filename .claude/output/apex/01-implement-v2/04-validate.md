# Step 04: Validate

**Task:** Implémenter Orya v2 sur branche v2 selon SESSION_CONTEXT.md
**Started:** 2026-05-19T11:28:38Z

---

## Validation Progress

_Validation results will be appended here..._



## Validation runs (in sandbox)

| Check | Result |
|-------|--------|
| `python3 -m py_compile` on all 30 Python files (agent/, graphiti-server/) | ✅ exit=0 |
| `bash -n server/scripts/init-postgres.sh` | ✅ valid |
| `bun install` in `server/gateway` | ✅ 6 packages, lockfile updated |
| `tsc --noEmit -p server/gateway/tsconfig.json` | ✅ exit=0 |
| `bun install` at root + `tsc --noEmit` (CLI) | ✅ exit=0 |
| Module load: `models.entities`, `models.schemas`, `models.state`, `persona.*`, `settings` | ✅ Pydantic validation works (`ChatRequest.opt_in_response.decision == 'accept'`) |
| Unit: `extract_quick_node` on 'je m\'appelle Bob et j\'habite à Lyon' | ✅ extracts `{name: Bob}`, `{city: Lyon}` (and a 'need' false-positive — acceptable for rule-based) |
| Unit: `extract_quick_node` on 'je suis dev et j\'ai 32 ans' | ✅ extracts `{occupation: true, age: 32}` |
| Unit: `make_detect_intent_node(None)` heuristic on 'je cherche un dev front à Lyon' | ✅ returns `{action: search, domain: software_dev, location: Lyon}` |
| Unit: heuristic on 'salut' (short) | ✅ `{action: chat}` |
| Unit: heuristic on a long unrelated sentence with no LLM fallback | ✅ defaults to `{action: chat}` (graceful) |

## What was NOT tested live

The sandbox has Python 3.9, so `graphiti-core` (>=3.10) cannot install here. Every module that does `from graphiti_core import …` (graph.py, main.py, providers/graphiti_client.py, nodes/persist_episode.py, nodes/retrieve_context.py, nodes/search_match.py) was syntax-validated via `py_compile` but not imported. They will be exercised at container boot in production (Python 3.12).

LLM and HuggingFace endpoints are not reachable from the sandbox without API keys, so:
- `provider/llm_router.py.build_llm` was not invoked.
- `Graphiti.add_episode` / `search` were not called.
- `AsyncPostgresSaver.setup()` was not invoked.

## Smoke test commands for production

```bash
# 1. Build
cd server && cp .env.example .env  # edit secrets first
docker build -t orya:v2 .

# 2. Run with persistent volumes
docker run -d --name orya \
    -p 4001:4001 -p 8000:8000 \
    -v orya_data:/data \
    -v orya_pg:/var/lib/postgresql/data \
    --env-file .env \
    orya:v2

# 3. Wait ~30s, then probe
curl -s http://localhost:4001/health
curl -s http://localhost:5001/health   # if exposed
curl -s http://localhost:8000/health   # if exposed

# 4. Try the CLI
SANDBOX_WS=ws://localhost:4001/ws bun /path/to/orya_cli/orya.ts
```
