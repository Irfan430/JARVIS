# 🤖 J.A.R.V.I.S — Development Plan

> **Just A Rather Very Intelligent System**
> A modular, agent-first personal AI assistant.

---

## Architecture

```
User (Telegram) → Bot Interface → Orchestrator → Planner (LLM) → Executor → Tool Router → Tools
                                         ↓
                                   Memory (STM + LTM)
                                         ↓
                                   LLM Providers (MiMo, OpenAI, Claude)
```

## Phases

### Phase 1: Core Foundation ✅
- [x] Orchestrator — session management, query routing
- [x] Planner — LLM-based task decomposition
- [x] Executor — step-by-step execution engine
- [x] Tool Router — dynamic tool dispatch

### Phase 2: Tools ✅
- [x] Search — DuckDuckGo web search
- [x] Calculator — safe math evaluation
- [x] Shell — terminal command execution
- [x] File — read/write/list/search
- [x] Browser — HTTP web scraping
- [x] API — generic REST client
- [x] Python — sandboxed code execution

### Phase 3: Memory ✅
- [x] STM — sliding window conversation memory
- [x] LTM — SQLite persistent fact storage
- [x] Manager — auto-extract facts, context enrichment

### Phase 4: LLM Providers ✅
- [x] OpenAI — GPT-4o/5
- [x] MiMo — Xiaomi MiMo
- [x] Anthropic — Claude
- [x] Manager — fallback chain, load balancing

### Phase 5: Voice I/O ✅
- [x] STT — OpenAI Whisper
- [x] TTS — edge-tts (free, high-quality)
- [x] Manager — unified voice pipeline

### Phase 6: Interface ✅
- [x] Telegram Bot — 12 commands, voice handler
- [x] Security — auth, rate limiting, sanitization
- [x] Config — Pydantic settings, .env + YAML

### Phase 7: Testing 🔄
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] E2E tests

### Phase 8: Documentation 🔄
- [x] README
- [x] Architecture docs
- [x] Setup guide
- [ ] Contributing guide

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Bot | python-telegram-bot |
| AI | MiMo, OpenAI, Claude |
| Memory | SQLite (LTM), In-memory (STM) |
| Voice | Whisper STT, edge-tts |
| Search | duckduckgo-search |
| HTTP | httpx |
| Config | Pydantic + YAML |
| Testing | pytest |

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize JARVIS |
| `/help` | List all commands |
| `/chat <msg>` | Chat with AI |
| `/ask <question>` | Ask a question |
| `/search <query>` | Web search |
| `/download <url>` | Download media |
| `/voice` | Voice interaction |
| `/model` | Switch AI model |
| `/clear` | Clear conversation |
| `/status` | System status |
| `/ping` | Latency check |
| `/sys` | System info |

---
*Built by Irfan Ahmmed — https://github.com/irfan420x*
