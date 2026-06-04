# Architecture

## Overview

JARVIS follows a **modular agent-first architecture** where the Orchestrator coordinates between specialized components.

## Components

### Orchestrator
Central coordinator. Receives user queries, manages sessions, calls Planner, routes results.

### Planner
LLM-powered task decomposition. Takes a query + context, returns a structured plan of tool calls.

### Executor
Step-by-step execution engine. Takes plan steps, routes to tools, collects results, handles failures.

### Tools
7 built-in tools: Search, Calculator, Shell, File, Browser, API, Python. Each inherits from BaseTool.

### Memory
- **STM** (Short-Term): In-memory sliding window for conversation context
- **LTM** (Long-Term): SQLite-based persistent storage for facts and preferences

### Providers
Multi-LLM support: MiMo, OpenAI, Anthropic. Fallback chain for reliability.

### Voice
STT (Whisper) + TTS (edge-tts) for voice interaction.

### Security
Authentication, rate limiting (token bucket), input sanitization.

## Data Flow

```
1. User sends message via Telegram
2. Bot interface receives → sanitizes → rate-checks
3. Orchestrator loads session context (STM)
4. Planner calls LLM → decomposes into steps
5. Executor runs each step via Tool Router
6. Tool Router dispatches to correct tool
7. Results collected → formatted → sent back
8. Context saved to STM, important facts to LTM
```

## Design Principles

- **Modularity**: Each component is independent, testable
- **Async**: All I/O operations are async
- **Fail-safe**: Tool failures don't crash the system
- **Extensible**: Add new tools by inheriting BaseTool
- **Observable**: All actions are logged
