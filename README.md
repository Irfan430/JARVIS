<div align="center">

```
    ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
    ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
    ██║███████║██████╔╝██║   ██║██║███████╗
██  ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
███████║██║  ██║██║  ██║ ╚████╔╝ ██║███████║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝ ╚═╝╚══════╝
```

### Just A Rather Very Intelligent System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-26A5E4?style=flat-square&logo=telegram&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00ff88?style=flat-square)
![Version](https://img.shields.io/badge/Version-0.1.0-blue?style=flat-square)

**A modular, agent-first personal AI assistant.**
Voice, memory, multi-LLM, tools — all from Telegram.

[Quick Start](#-quick-start) • [Features](#-features) • [Architecture](#-architecture) • [Commands](#-commands) • [Roadmap](#-roadmap)

</div>

---

## Features

| Category | What it does |
|----------|-------------|
| 🧠 **Multi-LLM** | Chat with MiMo, GPT-4o, Claude — switch on the fly |
| 🗣️ **Voice I/O** | Speak to JARVIS (Whisper STT), hear replies (edge-tts) |
| 🌐 **Web Search** | Search the web, summarize URLs, read RSS |
| 📥 **Media DL** | Download YouTube, TikTok, Instagram, Twitter |
| 💾 **Memory** | Remembers conversations (STM) + facts (LTM) |
| 🔧 **7 Tools** | Search, Calculator, Shell, File, Browser, API, Python |
| 🔒 **Security** | Auth, rate limiting, input sanitization |
| 📊 **Monitoring** | System status, CPU/RAM/Disk alerts |

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Telegram Bot      │
                    │  (python-tg-bot)    │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │     Orchestrator     │
                    │  (Session Manager)  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │    Planner Agent     │
                    │  (LLM Task Decompose)│
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Execution Engine   │
                    │  (Step-by-Step Run)  │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼─────┐ ┌──────▼──────┐ ┌──────▼──────┐
     │    Tools     │ │   Memory    │ │   Voice     │
     │ Search/Calc  │ │ STM + LTM   │ │ STT + TTS   │
     │ Shell/File   │ │ SQLite      │ │ Whisper     │
     │ Browser/API  │ │             │ │ edge-tts    │
     └──────────────┘ └─────────────┘ └─────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/irfan420x/JARVIS.git
cd JARVIS

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — add your Telegram bot token + API keys

# 4. Run
python -m src.main
```

### Prerequisites

- Python 3.11+
- Telegram Bot Token ([@BotFather](https://t.me/BotFather))
- At least one LLM API key (MiMo recommended — free tier available)

---

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Initialize JARVIS | `/start` |
| `/help` | List all commands | `/help` |
| `/chat <msg>` | Chat with AI | `/chat What's the weather?` |
| `/ask <q>` | Ask a question | `/ask Explain quantum computing` |
| `/search <q>` | Web search | `/search Python async tutorial` |
| `/download <url>` | Download media | `/download https://youtube.com/watch?v=...` |
| `/voice` | Voice mode | `/voice` (then send audio) |
| `/model` | Switch AI model | `/model` |
| `/clear` | Clear conversation | `/clear` |
| `/status` | System status | `/status` |
| `/ping` | Latency check | `/ping` |
| `/sys` | System info | `/sys` |

---

## Tools

JARVIS has 7 built-in tools that the AI can use automatically:

| Tool | What it does |
|------|-------------|
| `search` | DuckDuckGo web search |
| `calculator` | Safe math evaluation |
| `shell` | Execute terminal commands |
| `file` | Read/write/list files |
| `browser` | Fetch & scrape web pages |
| `api` | Generic REST API calls |
| `python` | Execute Python code |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11+ |
| **Bot** | python-telegram-bot |
| **LLM** | MiMo, OpenAI, Anthropic |
| **Voice STT** | OpenAI Whisper |
| **Voice TTS** | edge-tts (free) |
| **Search** | duckduckgo-search |
| **HTTP** | httpx (async) |
| **Memory** | SQLite + in-memory |
| **Config** | Pydantic + YAML |
| **Testing** | pytest |

---

## Roadmap

- [x] **v0.1** — Core orchestrator, planner, tools, memory
- [x] **v0.1** — Telegram bot interface
- [x] **v0.1** — Voice I/O (Whisper + edge-tts)
- [ ] **v0.2** — Browser automation (Playwright)
- [ ] **v0.2** — Media downloader (yt-dlp)
- [ ] **v0.3** — Long-term memory (Vector DB + RAG)
- [ ] **v1.0** — Multi-agent orchestration
- [ ] **v1.0** — CI/CD + Docker deployment
- [ ] **v1.0** — Web dashboard

---

## Project Structure

```
JARVIS/
├── src/
│   ├── orchestrator/    # Core session management
│   ├── planner/         # LLM task decomposition
│   ├── executor/        # Step execution engine
│   ├── tools/           # 7 built-in tools
│   ├── memory/          # STM + LTM
│   ├── providers/       # Multi-LLM support
│   ├── voice/           # STT + TTS
│   ├── interfaces/      # Telegram bot
│   └── security/        # Auth + rate limiting
├── config/              # Settings
├── tests/               # Unit + integration
├── docs/                # Documentation
└── data/                # Runtime data
```

---

## Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit (`git commit -m 'feat: add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Open Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by [Irfan Ahmmed](https://github.com/irfan420x)**

*"The best way to predict the future is to invent it."*

</div>
