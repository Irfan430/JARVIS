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
    
    # ---- Initialize Telegram Bot ----
    from src.interfaces.telegram_bot import TelegramBot
    
    telegram_bot = TelegramBot(
        config=config,
        authenticator=authenticator,
        rate_limiter=rate_limiter,
        sanitizer=sanitizer,
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
