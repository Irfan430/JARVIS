"""
JARVIS Telegram Bot Interface
Full-featured Telegram bot with command handling, voice support,
inline buttons, progress indicators, and error handling.
"""
from pathlib import Path

import asyncio
import time
import traceback
from typing import Optional, Any, Dict, Callable, Awaitable
from functools import wraps

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode, ChatAction
from loguru import logger


# ============================================================
# Rate limit decorator for handlers
# ============================================================

def rate_limited(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator to apply rate limiting to a handler function."""
    @wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        bot = context.bot_data.get("jarvis_bot")
        if bot and bot.rate_limiter:
            user_id = update.effective_user.id
            if bot.rate_limiter.is_rate_limited(user_id):
                remaining = bot.rate_limiter.get_reset_time(user_id)
                await update.message.reply_text(
                    f"⏳ Rate limited. Try again in {remaining:.0f}s."
                )
                return
        return await func(self, update, context, *args, **kwargs)
    return wrapper


# ============================================================
# Main Telegram Bot Class
# ============================================================

class TelegramBot:
    """
    Full-featured Telegram bot for JARVIS AI Assistant.
    
    Features:
    - Command handlers: /start, /help, /ping, /chat, /ask, /search,
                        /download, /voice, /model, /clear, /status, /sys
    - Free text message handling
    - Voice message handling (STT)
    - Inline button callbacks
    - Progress indicators
    - Error handling
    - Per-user rate limiting
    """

    def __init__(self, config, authenticator=None, rate_limiter=None, sanitizer=None,
                 memory=None, voice=None, tool_router=None):
        """
        Initialize the Telegram bot.
        
        Args:
            config: AppConfig instance with telegram settings.
            authenticator: Authenticator instance (optional).
            rate_limiter: RateLimiter instance (optional).
            sanitizer: InputSanitizer instance (optional).
            memory: MemoryManager instance (optional).
            voice: VoiceManager instance (optional).
            tool_router: ToolRouter instance (optional).
        """
        self.config = config
        self.auth = authenticator
        self.rate_limiter = rate_limiter
        self.sanitizer = sanitizer
        self.memory = memory
        self.voice = voice
        self.tool_router = tool_router
        self.app: Optional[Application] = None
        self._handlers_registered = False
        
        # Per-user conversation context (fallback if memory not available)
        self._conversations: Dict[int, list] = {}
        
        # Per-user model selection
        self._user_models: Dict[int, str] = {}
        
        # Per-user voice mode
        self._voice_mode: set = set()
        
        # Start time
        self._start_time: Optional[float] = None
        
        logger.info("TelegramBot initialized")
        if self.memory:
            logger.info("  ✓ Memory (STM+LTM) connected")
        if self.voice:
            logger.info("  ✓ Voice (STT+TTS) connected")
        if self.tool_router:
            logger.info("  ✓ Tool router connected")

    def _owner_only(func):
        """Decorator for owner-only commands."""
        @wraps(func)
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            bot = context.bot_data.get("jarvis_bot")
            if bot and bot.auth:
                user_id = update.effective_user.id
                if not bot.auth.is_owner(user_id):
                    await update.message.reply_text("⛔ Owner-only command.")
                    return
            return await func(self, update, context, *args, **kwargs)
        return wrapper

    def _authorized_only(func):
        """Decorator for authorized-user-only commands."""
        @wraps(func)
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            bot = context.bot_data.get("jarvis_bot")
            if bot and bot.auth:
                user_id = update.effective_user.id
                if not bot.auth.is_authorized(user_id):
                    await update.message.reply_text(
                        "⛔ You are not authorized to use this bot.\n"
                        "Contact the bot owner for access."
                    )
                    return
            return await func(self, update, context, *args, **kwargs)
        return wrapper

    async def _send_typing(self, update: Update, duration: float = 0.5):
        """Send typing indicator."""
        try:
            await update.message.chat.send_action(ChatAction.TYPING)
        except Exception:
            pass
        if duration > 0:
            await asyncio.sleep(duration)

    async def _edit_with_retry(
        self, message, text: str, parse_mode: str = ParseMode.HTML
    ) -> bool:
        """Edit a message with retry on flood wait."""
        try:
            await message.edit_text(text, parse_mode=parse_mode)
            return True
        except Exception:
            try:
                await message.edit_text(text, parse_mode=None)
                return True
            except Exception:
                return False

    # --------------------------------------------------------
    # Command Handlers
    # --------------------------------------------------------

    @_authorized_only
    @rate_limited
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        keyboard = [
            [
                InlineKeyboardButton("📖 Help", callback_data="help"),
                InlineKeyboardButton("📊 Status", callback_data="status"),
            ],
            [
                InlineKeyboardButton("💬 Chat", callback_data="chat_start"),
                InlineKeyboardButton("🔍 Search", callback_data="search_start"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome = (
            f"👋 Welcome, <b>{user.first_name}</b>!\n\n"
            "I am <b>JARVIS</b> — your personal AI assistant.\n\n"
            "Type a message to chat, or use these commands:\n"
            "• /help — List all commands\n"
            "• /chat &lt;message&gt; — Chat with me\n"
            "• /search &lt;query&gt; — Search the web\n"
            "• /voice — Voice mode\n"
            "• /status — System status\n\n"
            "Or just type anything! 🤖"
        )
        await update.message.reply_text(welcome, reply_markup=reply_markup)
        logger.info(f"User {user.id} ({user.username or user.first_name}) started bot")

    @_authorized_only
    @rate_limited
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = (
            "🤖 <b>JARVIS AI Assistant — Commands</b>\n\n"
            "<b>Chat:</b>\n"
            "  /chat &lt;message&gt; — Send a chat message\n"
            "  /ask &lt;question&gt; — Ask a specific question\n"
            "  /clear — Clear conversation history\n\n"
            "<b>Tools:</b>\n"
            "  /search &lt;query&gt; — Web search\n"
            "  /download &lt;url&gt; — Download a file\n"
            "  /voice — Toggle voice mode\n\n"
            "<b>System:</b>\n"
            "  /model [name] — View or change AI model\n"
            "  /status — Show system status\n"
            "  /ping — Latency check\n"
            "  /sys — System information\n\n"
            "<b>Tips:</b>\n"
            "• You can also just type a message to chat!\n"
            "• Send a voice message for speech-to-text\n"
            "• Use inline buttons for quick actions\n"
        )
        await update.message.reply_text(help_text)

    @_authorized_only
    @rate_limited
    async def cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ping command."""
        start = time.time()
        msg = await update.message.reply_text("🏓 Pong!")
        latency = (time.time() - start) * 1000
        await msg.edit_text(f"🏓 Pong! Latency: {latency:.0f}ms")

    @_authorized_only
    @rate_limited
    async def cmd_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chat command."""
        if not context.args:
            await update.message.reply_text(
                "Usage: /chat &lt;your message&gt;\n"
                "Example: /chat What is the weather today?"
            )
            return
        
        message = " ".join(context.args)
        await self._handle_chat_message(update, message, context)

    @_authorized_only
    @rate_limited
    async def cmd_ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ask command — same as /chat but emphasizes Q&A."""
        if not context.args:
            await update.message.reply_text(
                "Usage: /ask &lt;your question&gt;\n"
                "Example: /ask How does photosynthesis work?"
            )
            return
        
        message = " ".join(context.args)
        await self._handle_chat_message(update, message, context)

    @_authorized_only
    @rate_limited
    async def cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /search command."""
        if not context.args:
            await update.message.reply_text(
                "Usage: /search &lt;query&gt;\n"
                "Example: /search latest AI news"
            )
            return

        query = " ".join(context.args)
        
        # Sanitize
        if self.sanitizer:
            query = self.sanitizer.sanitize_command_arg(query)

        progress_msg = await update.message.reply_text("🔍 Searching...")
        
        try:
            # Use ToolRouter if available
            if self.tool_router:
                outcome = await self.tool_router.dispatch("search", {"query": query}, timeout=15.0)
                if hasattr(outcome, 'output'):
                    results_text = outcome.output
                    await progress_msg.edit_text(f"🔍 <b>Results for:</b> <i>{query}</i>\n\n{results_text[:3500]}")
                    return
                elif hasattr(outcome, 'error'):
                    await progress_msg.edit_text(f"❌ Search error: {outcome.error}")
                    return
            
            # Fallback to standalone search
            results = await self._standalone_search(query)
            
            if results:
                text = f"🔍 <b>Results for:</b> <i>{query}</i>\n\n"
                for i, result in enumerate(results[:5], 1):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No description")
                    url = result.get("url", "")
                    text += f"{i}. <b>{title}</b>\n"
                    text += f"   {snippet[:150]}...\n"
                    text += f"   <a href=\"{url}\">Link</a>\n\n"
            else:
                text = f"🔍 No results found for: <i>{query}</i>"
            
            await progress_msg.edit_text(text, parse_mode=ParseMode.HTML)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            await progress_msg.edit_text(f"❌ Search error: {str(e)}")

    async def _standalone_search(self, query: str) -> list:
        """Fallback search using duckduckgo-search."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                return results
        except Exception as e:
            logger.error(f"Standalone search failed: {e}")
            return []

    @_authorized_only
    @rate_limited
    async def cmd_download(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /download command."""
        if not context.args:
            await update.message.reply_text(
                "Usage: /download &lt;url&gt;\n"
                "Example: https://example.com/file.pdf"
            )
            return

        url = context.args[0]
        
        # Sanitize URL
        if self.sanitizer:
            try:
                url = self.sanitizer.sanitize_url(url)
            except ValueError as e:
                await update.message.reply_text(f"❌ Invalid URL: {e}")
                return

        progress_msg = await update.message.reply_text("📥 Downloading...")
        
        try:
            import httpx
            import aiofiles
            import os
            from pathlib import Path
            
            # Extract filename from URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or "download"
            if len(filename) > 100:
                filename = filename[:100]
            
            # Download with streaming
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        await progress_msg.edit_text(
                            f"❌ Download failed: HTTP {response.status_code}"
                        )
                        return
                    
                    total = int(response.headers.get("content-length", 0))
                    data_dir = Path(self.config.data_dir) / "downloads"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    filepath = data_dir / filename
                    
                    downloaded = 0
                    async with aiofiles.open(filepath, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            await f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = (downloaded / total) * 100
                                if downloaded % (total // 5 or 1) == 0:
                                    await progress_msg.edit_text(
                                        f"📥 Downloading: {pct:.0f}% "
                                        f"({downloaded}/{total} bytes)"
                                    )
            
            size_mb = downloaded / (1024 * 1024)
            await progress_msg.edit_text(
                f"✅ Download complete!\n\n"
                f"📄 File: <code>{filename}</code>\n"
                f"📦 Size: {size_mb:.2f} MB\n"
                f"📁 Path: <code>{filepath}</code>"
            )
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            await progress_msg.edit_text(f"❌ Download failed: {str(e)}")

    @_authorized_only
    @rate_limited
    async def cmd_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /voice command — toggle voice mode."""
        user_id = update.effective_user.id
        
        # Check if voice is enabled in config
        if not self.config.voice.enabled:
            await update.message.reply_text(
                "🔇 Voice mode is not enabled in the configuration."
            )
            return
        
        # Check if VoiceManager is available
        if not self.voice:
            await update.message.reply_text(
                "🔇 Voice module not loaded. Install edge-tts: pip install edge-tts"
            )
            return
        
        # Toggle voice mode
        if user_id in self._voice_mode:
            self._voice_mode.discard(user_id)
            new_state = False
        else:
            self._voice_mode.add(user_id)
            new_state = True
        icon = "🔊" if new_state else "🔇"
        text = (
            f"{icon} Voice mode <b>{'enabled' if new_state else 'disabled'}</b>.\n\n"
            f"{'Send me a voice message and I\'ll transcribe and respond with voice!' if new_state else 'Text mode activated.'}"
        )
        await update.message.reply_text(text)

    @staticmethod
    def _set_model_keyboard():
        """Create model selection keyboard."""
        models = [
            ("GPT-4o", "gpt-4o"),
            ("GPT-4o Mini", "gpt-4o-mini"),
            ("Claude 3.5 Sonnet", "claude-3-5-sonnet"),
            ("Claude 3 Haiku", "claude-3-haiku"),
        ]
        keyboard = []
        for name, model_id in models:
            keyboard.append([
                InlineKeyboardButton(name, callback_data=f"set_model:{model_id}")
            ])
        keyboard.append([
            InlineKeyboardButton("❌ Cancel", callback_data="cancel")
        ])
        return InlineKeyboardMarkup(keyboard)

    @_authorized_only
    @rate_limited
    async def cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /model command — view or change AI model."""
        user_id = update.effective_user.id
        current = self._user_models.get(user_id, self.config.ai.model)
        
        if context.args:
            # Set model directly
            new_model = context.args[0]
            self._user_models[user_id] = new_model
            await update.message.reply_text(
                f"🤖 Model changed to: <b>{new_model}</b>"
            )
        else:
            # Show selection keyboard
            text = (
                f"🤖 Current model: <b>{current}</b>\n\n"
                f"Select a new model:"
            )
            await update.message.reply_text(
                text, reply_markup=self._set_model_keyboard()
            )

    @_authorized_only
    @rate_limited
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command — clear conversation history."""
        user_id = update.effective_user.id
        if self.memory:
            self.memory.new_session(session_id=str(user_id))
        if user_id in self._conversations:
            self._conversations[user_id] = []
        context.user_data.clear()
        await update.message.reply_text("🧹 Conversation history cleared.")

    @_authorized_only
    @rate_limited
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command — system status."""
        user_id = update.effective_user.id
        
        # Gather stats
        rate_info = ""
        if self.rate_limiter:
            status = self.rate_limiter.get_user_status(user_id)
            rate_info = (
                f"📊 Rate limit: {status['remaining']:.0f}/{status['capacity']} "
                f"(resets in {status['reset_in']:.0f}s)"
            )
        
        conv_count = len(self._conversations.get(user_id, []))
        current_model = self._user_models.get(user_id, self.config.ai.model)
        
        status_text = (
            "📊 <b>JARVIS System Status</b>\n\n"
            f"🤖 Model: <code>{current_model}</code>\n"
            f"💬 Conversation: {conv_count} messages\n"
            f"🔐 User ID: <code>{user_id}</code>\n"
            f"👑 Owner: {'Yes' if self.auth and self.auth.is_owner(user_id) else 'No'}\n"
        )
        
        # Memory stats
        if self.memory:
            try:
                stats = self.memory.stats()
                status_text += f"\n💾 Memory: STM={stats.get('stm_messages', 0)} msgs, LTM={stats.get('ltm_entries', 0)} facts\n"
            except Exception:
                pass
        
        # Voice status
        if self.voice:
            try:
                avail = self.voice.is_available
                status_text += f"🗣️ Voice: STT={'✓' if avail.get('stt') else '✗'}, TTS={'✓' if avail.get('tts') else '✗'}\n"
            except Exception:
                pass
        
        # Tools status
        if self.tool_router:
            try:
                tools = self.tool_router.list_tools()
                status_text += f"🔧 Tools: {len(tools)} registered ({', '.join(t.name for t in tools[:5])})\n"
            except Exception:
                pass
        
        if rate_info:
            status_text += f"\n{rate_info}\n"
        
        status_text += f"\n⏱️ Uptime: {self._get_uptime()}"
        
        await update.message.reply_text(status_text)

    def _get_uptime(self) -> str:
        """Get formatted uptime string."""
        if hasattr(self, '_start_time'):
            elapsed = time.time() - self._start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours}h {minutes}m {seconds}s"
        return "unknown"

    @_owner_only
    @rate_limited
    async def cmd_sys(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sys command — system info (owner only)."""
        import platform
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        sys_text = (
            "🖥 <b>System Information</b>\n\n"
            f"<b>OS:</b> {platform.system()} {platform.release()}\n"
            f"<b>Python:</b> {platform.python_version()}\n"
            f"<b>Machine:</b> {platform.machine()}\n\n"
            f"<b>CPU:</b> {cpu_percent}%\n"
            f"<b>Memory:</b> {memory.percent}% "
            f"({memory.used // (1024**2)}MB / {memory.total // (1024**2)}MB)\n"
            f"<b>Disk:</b> {disk.percent}% "
            f"({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)\n\n"
            f"<b>Active sessions:</b> {len(self._conversations)}\n"
        )
        
        await update.message.reply_text(sys_text)

    # --------------------------------------------------------
    # Message Handlers
    # --------------------------------------------------------

    @_authorized_only
    @rate_limited
    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle free-form text messages."""
        message = update.message.text
        if not message:
            return
        
        # Sanitize input
        if self.sanitizer:
            try:
                message = self.sanitizer.sanitize_message(message)
            except ValueError as e:
                await update.message.reply_text(f"⚠️ Input rejected: {e}")
                return
        
        await self._handle_chat_message(update, message, context)

    async def _handle_chat_message(self, update: Update, message: str, context=None):
        """Core chat message handler — uses memory, tools, and voice."""
        user_id = update.effective_user.id
        
        # ---- Store user message in memory ----
        if self.memory:
            self.memory.add_message("user", message, metadata={"user_id": user_id})
        else:
            # Fallback: use _conversations dict
            if user_id not in self._conversations:
                self._conversations[user_id] = []
            self._conversations[user_id].append({"role": "user", "content": message})
            if len(self._conversations[user_id]) > 50:
                self._conversations[user_id] = self._conversations[user_id][-50:]
        
        # ---- Send typing indicator ----
        await self._send_typing(update, 1.0)
        
        progress_msg = await update.message.reply_text("🧠 Thinking...")
        
        try:
            # ---- Try tool execution first (if message looks like a tool request) ----
            tool_result = await self._try_tools(message, context)
            if tool_result:
                response = tool_result
            else:
                # ---- Use AI backend for chat ----
                bot_instance = context.bot_data.get("jarvis_bot") if context else self
                if bot_instance and hasattr(bot_instance, '_app') and bot_instance._app:
                    model = self._user_models.get(user_id, self.config.ai.model)
                    
                    # Build messages list from memory
                    if self.memory:
                        messages = self.memory.get_context(system_prompt=self.config.ai.system_prompt)
                    else:
                        messages = []
                        if self.config.ai.system_prompt:
                            messages.append({"role": "system", "content": self.config.ai.system_prompt})
                        messages.extend([
                            {"role": m["role"], "content": m["content"]}
                            for m in self._conversations.get(user_id, [])
                        ])
                    
                    response = await bot_instance._app.chat(
                        messages=messages,
                        model=model,
                    )
                    # Extract content from ChatResponse object
                    response = response.content if hasattr(response, 'content') else str(response)
                else:
                    response = (
                        "🤖 I'm in standalone mode. The AI backend is not connected.\n\n"
                        "To connect the AI provider, make sure the full JARVIS "
                        "application is properly initialized.\n\n"
                        f"Your message was: <i>{message[:200]}</i>"
                    )
            
            # ---- Store assistant response in memory ----
            if self.memory:
                self.memory.add_message("assistant", response, metadata={"user_id": user_id})
            else:
                if user_id not in self._conversations:
                    self._conversations[user_id] = []
                self._conversations[user_id].append({"role": "assistant", "content": response})
            
            # ---- Voice mode: send audio response ----
            voice_enabled = user_id in self._voice_mode
            if voice_enabled and self.voice:
                try:
                    audio_path = await self.voice.speak(response)
                    if audio_path and Path(audio_path).exists():
                        from telegram import InputFile
                        with open(audio_path, 'rb') as af:
                            await update.message.reply_voice(voice=af)
                except Exception as ve:
                    logger.warning(f"TTS failed: {ve}")
            
            # ---- Send text response (skip if voice was sent) ----
            if len(response) > 4000:
                parts = self._split_message(response, 4000)
                await progress_msg.edit_text(parts[0])
                for part in parts[1:]:
                    await update.message.reply_text(part)
            else:
                await progress_msg.edit_text(response)
                
        except Exception as e:
            logger.error(f"Chat error for user {user_id}: {e}\n{traceback.format_exc()}")
            try:
                await progress_msg.edit_text(
                    f"❌ Error processing message:\n<code>{str(e)[:500]}</code>"
                )
            except Exception:
                pass

    def _split_message(self, text: str, max_len: int = 4000) -> list:
        """Split a long message into parts."""
        parts = []
        while len(text) > max_len:
            # Find a good break point
            idx = text.rfind("\n", 0, max_len)
            if idx == -1:
                idx = text.rfind(". ", 0, max_len)
            if idx == -1:
                idx = text.rfind(" ", 0, max_len)
            if idx == -1:
                idx = max_len
            
            parts.append(text[:idx])
            text = text[idx:].lstrip()
        if text:
            parts.append(text)
        return parts

    async def _try_tools(self, message: str, context=None) -> Optional[str]:
        """Try to execute message as a tool command. Returns result or None."""
        if not self.tool_router:
            return None
        
        msg = message.strip().lower()
        
        # Simple pattern matching for tool invocation
        tool_map = {
            ("calculate", "calc", "math"): ("calculator", {"expression": message}),
            ("search", "find", "look up"): ("search", {"query": message}),
            ("run python", "execute python", "python code"): ("python", {"code": message}),
            ("list files", "ls", "dir"): ("file", {"action": "list", "path": "."}),
            ("read file", "cat"): ("file", {"action": "read", "path": message.split(maxsplit=1)[-1] if " " in message else "."}),
            ("system info", "sysinfo"): ("shell", {"command": "uname -a && free -h && df -h"}),
        }
        
        for patterns, (tool_name, params) in tool_map.items():
            if any(msg.startswith(p) or msg == p for p in patterns):
                try:
                    outcome = await self.tool_router.dispatch(tool_name, params, timeout=30.0)
                    if hasattr(outcome, 'output'):
                        return f"🔧 <b>{tool_name}</b> result:\n<pre>{outcome.output[:3000]}</pre>"
                    elif hasattr(outcome, 'error'):
                        return f"❌ Tool error: {outcome.error}"
                except Exception as e:
                    logger.warning(f"Tool dispatch error: {e}")
                    return f"❌ Tool error: {str(e)[:200]}"
        
        return None

    @_authorized_only
    async def handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages — STT via VoiceManager or fallback whisper."""
        if not self.config.voice.enabled:
            await update.message.reply_text("🔇 Voice mode is not enabled.")
            return
        
        progress_msg = await update.message.reply_text("🎙 Transcribing voice message...")
        
        try:
            # Download the voice file
            tg_voice = update.message.voice or update.message.audio
            if not tg_voice:
                await progress_msg.edit_text("❌ No voice data found.")
                return
            
            file = await context.bot.get_file(tg_voice.file_id)
            import tempfile
            import os
            
            suffix = ".ogg" if update.message.voice else ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name
            
            # Use VoiceManager if available, else fallback
            transcript = None
            if self.voice:
                try:
                    result = await self.voice.listen(tmp_path)
                    transcript = result.get("text", "") if isinstance(result, dict) else str(result)
                except Exception as ve:
                    logger.warning(f"VoiceManager STT failed: {ve}, falling back to whisper")
                    transcript = await self._transcribe_audio(tmp_path)
            else:
                transcript = await self._transcribe_audio(tmp_path)
            
            # Cleanup
            os.unlink(tmp_path)
            
            if transcript:
                await progress_msg.edit_text(
                    f"🎙 <b>Transcribed:</b>\n\n{transcript}\n\n"
                    f"<i>Reply to start a conversation about this.</i>"
                )
                # Auto-respond with chat
                await self._handle_chat_message(update, transcript, context)
            else:
                await progress_msg.edit_text("❌ Could not transcribe audio.")
                
        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            await progress_msg.edit_text(f"❌ Transcription error: {str(e)}")

    async def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file to text."""
        try:
            import whisper
            import os
            
            model_size = self.config.voice.stt_model
            logger.info(f"Loading whisper model: {model_size}")
            
            model = whisper.load_model(model_size)
            result = model.transcribe(audio_path)
            
            return result.get("text", "").strip()
            
        except ImportError:
            logger.warning("Whisper not installed. Using placeholder.")
            return "[Voice transcription unavailable — install openai-whisper]"
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return None

    # --------------------------------------------------------
    # Callback Query Handler
    # --------------------------------------------------------

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callback queries."""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = query.from_user.id
        
        if data == "help":
            help_text = (
                "📖 <b>JARVIS AI Assistant — Commands</b>\n\n"
                "• /chat — Chat with me\n"
                "• /search — Search the web\n"
                "• /voice — Voice mode\n"
                "• /status — System status\n"
                "• /help — Show this help\n"
            )
            await query.edit_message_text(help_text, parse_mode=ParseMode.HTML)
            
        elif data == "status":
            current_model = self._user_models.get(user_id, self.config.ai.model)
            status_text = (
                f"📊 <b>Status</b>\n\n"
                f"🤖 Model: <code>{current_model}</code>\n"
                f"💬 Messages: {len(self._conversations.get(user_id, []))}\n"
            )
            await query.edit_message_text(status_text, parse_mode=ParseMode.HTML)
            
        elif data == "chat_start":
            await query.edit_message_text(
                "💬 <b>Chat Mode</b>\n\n"
                "Just type your message below!\n"
                "Or use /chat &lt;message&gt;"
            )
            
        elif data == "search_start":
            await query.edit_message_text(
                "🔍 <b>Search Mode</b>\n\n"
                "Type /search &lt;your query&gt; to search the web."
            )
            
        elif data.startswith("set_model:"):
            model_id = data.split(":", 1)[1]
            self._user_models[user_id] = model_id
            await query.edit_message_text(
                f"🤖 Model changed to: <b>{model_id}</b>",
                parse_mode=ParseMode.HTML,
            )
            
        elif data == "cancel":
            await query.edit_message_text("❌ Cancelled.")
            
        else:
            await query.edit_message_text(f"Unknown action: {data}")

    # --------------------------------------------------------
    # Error Handler
    # --------------------------------------------------------

    async def handle_error(self, update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
        """Global error handler."""
        error = context.error
        logger.error(f"Bot error: {error}\n{traceback.format_exc()}")
        
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "⚠️ An internal error occurred. Please try again.\n"
                    f"<code>{str(error)[:200]}</code>"
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # Bot Lifecycle
    # --------------------------------------------------------

    def _register_handlers(self):
        """Register all command and message handlers."""
        if self._handlers_registered:
            return
        
        # Command handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("ping", self.cmd_ping))
        self.app.add_handler(CommandHandler("chat", self.cmd_chat))
        self.app.add_handler(CommandHandler("ask", self.cmd_ask))
        self.app.add_handler(CommandHandler("search", self.cmd_search))
        self.app.add_handler(CommandHandler("download", self.cmd_download))
        self.app.add_handler(CommandHandler("voice", self.cmd_voice))
        self.app.add_handler(CommandHandler("model", self.cmd_model))
        self.app.add_handler(CommandHandler("clear", self.cmd_clear))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("sys", self.cmd_sys))
        
        # Voice message handler
        self.app.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self.handle_voice_message)
        )
        
        # Text message handler (non-command)
        self.app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND, 
                self.handle_text_message
            )
        )
        
        # Callback query handler
        self.app.add_handler(CallbackQueryHandler(self.handle_callback_query))
        
        # Error handler
        self.app.add_error_handler(self.handle_error)
        
        self._handlers_registered = True
        logger.info("All handlers registered")

    async def _set_bot_commands(self):
        """Set bot command menu in Telegram."""
        commands = [
            BotCommand("start", "Start the bot"),
            BotCommand("help", "Show available commands"),
            BotCommand("chat", "Chat with JARVIS"),
            BotCommand("ask", "Ask a question"),
            BotCommand("search", "Search the web"),
            BotCommand("download", "Download a file"),
            BotCommand("voice", "Toggle voice mode"),
            BotCommand("model", "Change AI model"),
            BotCommand("clear", "Clear conversation"),
            BotCommand("status", "System status"),
            BotCommand("ping", "Check latency"),
            BotCommand("sys", "System info (owner)"),
        ]
        try:
            await self.app.bot.set_my_commands(commands)
            logger.info("Bot commands menu set")
        except Exception as e:
            logger.warning(f"Could not set bot commands: {e}")

    def build(self) -> Application:
        """Build and configure the Application."""
        builder = (
            Application.builder()
            .token(self.config.telegram.bot_token)
            .build()
        )
        
        self.app = builder
        
        # Store reference to ourselves in bot_data
        self.app.bot_data["jarvis_bot"] = self
        
        # Register handlers
        self._register_handlers()
        
        # Record start time
        self._start_time = time.time()
        
        logger.info("Telegram bot built successfully")
        return self.app

    async def run(self):
        """Start the bot (blocking)."""
        if not self.app:
            self.build()
        
        await self._set_bot_commands()
        
        logger.info("Starting Telegram bot...")
        
        # Use polling (webhook mode would be set up separately)
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        
        logger.info("🤖 JARVIS Telegram bot is online!")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down Telegram bot...")
        if self.app:
            if self.app.updater and self.app.updater.running:
                await self.app.updater.stop()
            if self.app.running:
                await self.app.stop()
            await self.app.shutdown()
        logger.info("Telegram bot shut down")
