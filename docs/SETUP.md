# Setup Guide

## Prerequisites

- Python 3.11 or higher
- pip
- git
- Telegram Bot Token

## Installation

```bash
# Clone
git clone https://github.com/irfan420x/JARVIS.git
cd JARVIS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

```bash
# Copy env template
cp .env.example .env

# Edit .env with your keys:
# TELEGRAM_BOT_TOKEN=your_token_from_botfather
# MIMO_API_KEY=your_mimo_key (free at xiaomimimo.com)
# OPENAI_API_KEY=your_openai_key (optional)
# ANTHROPIC_API_KEY=your_claude_key (optional)
```

## Running

```bash
# Start JARVIS
python -m src.main

# Or with specific config
JARVIS_CONFIG=config.yaml python -m src.main
```

## Getting API Keys

### Telegram Bot Token
1. Open Telegram, search for @BotFather
2. Send `/newbot`
3. Follow instructions
4. Copy the token

### MiMo API Key (Free)
1. Visit https://api.xiaomimimo.com
2. Sign up
3. Create API key

### OpenAI Key (Optional)
1. Visit https://platform.openai.com
2. Create API key

### Anthropic Key (Optional)
1. Visit https://console.anthropic.com
2. Create API key

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/unit/test_tools.py
```
