"""
JARVIS AI Assistant - Main Entry Point
Initializes all components and starts the bot.
"""

import sys
import signal
import asyncio
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO"):
    """Configure loguru logging."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        "logs/jarvis_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )


async def main():
    """Main application entry point."""
    # ---- Load configuration ----
    from config.settings import load_config

    logger.info("=" * 50)
    logger.info("  JARVIS AI Assistant — Starting up...")
    logger.info("=" * 50)

    config = load_config()
    setup_logging(config.log_level)

    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"AI provider: {config.ai.provider}")
    logger.info(f"Voice enabled: {config.voice.enabled}")

    # ---- Validate Telegram token ----
    if not config.telegram.bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set! Cannot start bot.")
        logger.error("Please set it in .env or environment variables.")
        sys.exit(1)

    # ---- Initialize Security ----
    from src.security.auth import Authenticator, SessionManager
    from src.security.rate_limiter import RateLimiter
    from src.security.sanitizer import InputSanitizer

    authenticator = Authenticator(
        owner_id=config.telegram.owner_id,
        allowed_users=set(config.telegram.allowed_users),
    )

    session_manager = SessionManager(timeout=config.security.session_timeout)

    rate_limiter = RateLimiter(
        max_requests=config.security.rate_limit_requests,
        window_seconds=config.security.rate_limit_window,
        cooldown_seconds=config.security.rate_limit_cooldown,
        enabled=config.security.rate_limit_enabled,
    )

    sanitizer = InputSanitizer(
        max_length=config.security.max_input_length,
        strict_mode=False,
    )

    logger.info("Security modules initialized")

    # ---- Initialize AI Backend ----
    ai_backend = None
    try:
        from src.providers import create_provider

        ai_backend = create_provider(
            provider=config.ai.provider,
            api_key=config.ai.api_key,
            model=config.ai.model,
            base_url=config.ai.base_url,
            system_prompt=config.ai.system_prompt,
        )
        logger.info(f"AI backend initialized: {config.ai.provider}/{config.ai.model}")
    except Exception as e:
        logger.warning(f"AI backend init failed: {e}")
        logger.info("Running in standalone mode (no AI backend)")

    # ---- Initialize Memory ----
    memory = None
    try:
        from src.memory.manager import MemoryManager
        memory = MemoryManager(
            stm_max_messages=50,
            auto_save_facts=True,
        )
        logger.info("Memory manager initialized (STM + LTM)")
    except Exception as e:
        logger.warning(f"Memory init failed: {e}")

    # ---- Initialize Voice ----
    voice = None
    if config.voice.enabled:
        try:
            from src.voice.manager import VoiceManager
            voice = VoiceManager(
                stt_model=config.voice.stt_model,
                tts_engine=config.voice.tts_provider,
                tts_voice=config.voice.tts_voice,
            )
            avail = voice.is_available
            logger.info(f"Voice manager initialized (STT: {avail.get('stt', False)}, TTS: {avail.get('tts', False)})")
        except Exception as e:
            logger.warning(f"Voice init failed: {e}")

    # ---- Initialize Tool Router ----
    tool_router = None
    try:
        from src.executor.tool_router import ToolRouter
        tool_router = ToolRouter(max_concurrent=5)

        # Register all 7 tools
        from src.tools.search_tool import SearchTool
        from src.tools.calculator_tool import CalculatorTool
        from src.tools.shell_tool import ShellTool
        from src.tools.file_tool import FileTool
        from src.tools.browser_tool import BrowserTool
        from src.tools.api_tool import APITool
        from src.tools.python_tool import PythonTool

        tools = [
            SearchTool(),
            CalculatorTool(),
            ShellTool(),
            FileTool(),
            BrowserTool(),
            APITool(),
            PythonTool(),
        ]

        for tool in tools:
            tool_router.register(
                name=tool.name,
                handler=tool.safe_execute,
                description=tool.description,
                parameters=tool.get_definition().get("parameters", {}),
                required_params=tool.required_params,
                tags=[tool.category] if hasattr(tool, 'category') else [],
            )
            logger.info(f"  Registered tool: {tool.name}")

        logger.info(f"Tool router initialized with {len(tools)} tools")
    except Exception as e:
        logger.warning(f"Tool router init failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Initialize Telegram Bot ----
    from src.interfaces.telegram_bot import TelegramBot

    telegram_bot = TelegramBot(
        config=config,
        authenticator=authenticator,
        rate_limiter=rate_limiter,
        sanitizer=sanitizer,
        memory=memory,
        voice=voice,
        tool_router=tool_router,
    )

    # Attach AI backend reference if available
    if ai_backend:
        telegram_bot._app = type("AIApp", (), {
            "chat": ai_backend.chat,
            "search": ai_backend.search if hasattr(ai_backend, 'search') else None,
        })()

    logger.info("Telegram bot interface initialized")

    # ---- Set up signal handlers ----
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.ensure_future(telegram_bot.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # ---- Start the bot ----
    logger.info("=" * 50)
    logger.info("  🤖 JARVIS is online and ready!")
    logger.info("=" * 50)

    try:
        await telegram_bot.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await telegram_bot.shutdown()
        logger.info("JARVIS shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
