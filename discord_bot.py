"""Discord bot — talking agent for project management.

Natural language interface in DMs:
- "create repo my-project" → creates GitHub repo
- "list my repos" → shows all repos
- "what's in vikunja" → lists projects + tasks
- "create task X in Y" → creates Vikunja task
- "start session for X" → new opencode session
- "what did we work on" → memory search + briefing
- Anything else → forward to opencode serve for coding help

Requires: discord.py 2.x, bot token in config.yaml under discord.token.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

import discord
from discord import app_commands

from github_ops import GitHubOps
from project_manager.registry import ProjectRegistry

if TYPE_CHECKING:
    from opencode_client import OpenCodeClient
    from memory import MemoryStore

logger = logging.getLogger("secretary.discord")


class SecretaryBot(discord.Client):
    """Discord bot that acts as a personal agent via DM."""

    def __init__(
        self,
        config: dict[str, Any],
        opencode: OpenCodeClient,
        memory: MemoryStore,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        super().__init__(intents=intents)

        self.config = config
        self.opencode = opencode
        self.memory = memory
        self.github = GitHubOps(
            default_owner=config.get("github", {}).get("owner", "0124212")
        )
        self.projects = ProjectRegistry(config)
        self.tree = app_commands.CommandTree(self)

        # Access control
        discord_cfg = config.get("discord", {})
        self.allowed_users: set[int] = set()
        for uid in discord_cfg.get("allowed_user_ids", []):
            try:
                self.allowed_users.add(int(uid))
            except (ValueError, TypeError):
                pass

        # Active DM sessions: user_id -> opencode session_id
        self._dm_sessions: dict[int, str] = {}

    async def setup_hook(self) -> None:
        self.tree.add_command(self.status_command)
        self.tree.add_command(self.briefing_command)
        self.tree.add_command(self.search_command)
        await self.tree.sync()
        logger.info("Slash commands synced")

    async def on_ready(self) -> None:
        logger.info("Discord bot logged in as %s (ID: %s)", self.user, self.user.id)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="DM me | repos, projects, sessions, coding",
            )
        )

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not isinstance(message.channel, discord.DMChannel):
            return

        user_id = message.author.id
        text = message.content.strip()

        # Access control
        if self.allowed_users and user_id not in self.allowed_users:
            await message.add_reaction("\u274c")
            await message.author.send(
                "You're not authorized. Ask the admin to add your user ID."
            )
            return

        if not text:
            return

        # Route by intent
        await message.add_reaction("\u23f3")
        try:
            reply = await self._route(text, user_id, message)
            await message.remove_reaction("\u23f3", self.user)
            if reply:
                await self._send_long(message.channel, reply)
        except Exception:
            await message.remove_reaction("\u23f3", self.user)
            await message.add_reaction("\u274c")
            logger.exception("Failed to process message from user %s", user_id)
            await message.author.send("Something went wrong. Try again.")

    # ── Intent routing ──────────────────────────────────────────

    async def _route(self, text: str, user_id: int, message: discord.Message) -> str:
        """Determine intent and dispatch to handler. Returns reply text."""
        lower = text.lower()

        # ── GitHub intents ──────────────────────────────────────
        if _matches(lower, r"\b(create|new|make)\b.*\b(repo|repository|github)\b"):
            return await self._handle_create_repo(text)
        if _matches(lower, r"\b(list|show|all|my)\b.*\b(repo|repos|repositories)\b"):
            return await self.github.list_repos()
        if _matches(lower, r"\b(repo|repos)\b.*\b(list|show)\b"):
            return await self.github.list_repos()
        if _matches(lower, r"\b(repo|repository)\b.*\b(info|details|about)\b"):
            return await self._handle_repo_info(text)
        if _matches(lower, r"\b(delete|remove)\b.*\b(repo|repository)\b"):
            return await self._handle_delete_repo(text)
        if _matches(lower, r"\b(clone)\b.*\b(repo)?\b"):
            return await self._handle_clone(text)
        if _matches(lower, r"\b(local|cloned)\b.*\b(repo|repos)\b"):
            return await self.github.list_local_repos()
        if _matches(lower, r"\b(search|find)\b.*\b(code|file)\b"):
            return await self._handle_search_code(text)
        if _matches(lower, r"\b(search|find)\b.*\b(repo|repos)\b"):
            return await self._handle_search_repos(text)
        if _matches(lower, r"\b(issue|issues|bug|bugs)\b.*\b(list|show)\b"):
            return await self._handle_list_issues(text)
        if _matches(lower, r"\b(create|new|open)\b.*\b(issue|bug)\b"):
            return await self._handle_create_issue(text)

        # ── Vikunja intents ─────────────────────────────────────
        if _matches(lower, r"\b(vikunja|project|projects)\b.*\b(list|show|all)\b"):
            return await self._handle_list_projects()
        if _matches(lower, r"\b(list|show|all)\b.*\b(vikunja|project|projects)\b"):
            return await self._handle_list_projects()
        if _matches(lower, r"\b(task|tasks)\b.*\b(list|show)\b"):
            return await self._handle_list_tasks(text)
        if _matches(lower, r"\b(create|new|add)\b.*\b(task)\b"):
            return await self._handle_create_task(text)
        if _matches(lower, r"\b(close|done|complete|finish)\b.*\b(task)\b"):
            return await self._handle_close_task(text)

        # ── Session intents ─────────────────────────────────────
        if _matches(lower, r"\b(start|new|create)\b.*\b(session|coding|work)\b"):
            return await self._handle_start_session(text, user_id)
        if _matches(lower, r"\b(session|sessions)\b.*\b(list|show)\b"):
            return await self._handle_list_sessions()
        if _matches(lower, r"\b(session|coding)\b.*\b(status|busy|idle)\b"):
            return await self._handle_session_status()

        # ── Memory intents ──────────────────────────────────────
        if _matches(lower, r"\b(what|how)\b.*\b(worked|did|done|recent)\b"):
            return await self._handle_memory_search(user_id)
        if _matches(lower, r"\b(search|find|recall|remember)\b"):
            return await self._handle_memory_search(user_id, text)
        if _matches(lower, r"\b(briefing|summary|status)\b"):
            return await self._handle_briefing()

        # ── Help ────────────────────────────────────────────────
        if _matches(lower, r"\b(help|commands?|what can)\b"):
            return self._help_text()

        # ── Default: forward to opencode ────────────────────────
        return await self._handle_opencode(text, user_id)

    # ── GitHub handlers ─────────────────────────────────────────

    async def _handle_create_repo(self, text: str) -> str:
        # Extract repo name: "create repo my-cool-project" or "create repo my cool project"
        m = re.search(r"(?:create|new|make)\s+(?:repo|repository|github)\s+[\"']?(\S+?)[\"']?\s*$", text, re.I)
        if not m:
            # Try to extract after "repo" keyword
            m = re.search(r"(?:repo|repository)\s+[\"']?([a-zA-Z0-9_-]+)", text, re.I)
        if not m:
            return "What should I name the repo? Usage: `create repo my-project`"
        name = m.group(1).strip("-")
        result = await self.github.create_repo(name, auto_init=True)
        # Save to memory
        self.memory.add(
            f"[fact] Created GitHub repo: {name}",
            metadata={"type": "extracted_fact", "action": "create_repo", "repo": name},
        )
        return result

    async def _handle_repo_info(self, text: str) -> str:
        m = re.search(r"(?:repo|repository)\s+(?:info|details|about)\s+(\S+)", text, re.I)
        if not m:
            return "Which repo? Usage: `repo info my-project`"
        return await self.github.get_repo(m.group(1))

    async def _handle_delete_repo(self, text: str) -> str:
        m = re.search(r"(?:delete|remove)\s+(?:repo|repository)\s+(\S+)", text, re.I)
        if not m:
            return "Which repo? Usage: `delete repo my-project`"
        return await self.github.delete_repo(m.group(1))

    async def _handle_clone(self, text: str) -> str:
        m = re.search(r"clone\s+(\S+)", text, re.I)
        if not m:
            return "Which repo? Usage: `clone my-project`"
        return await self.github.clone(m.group(1))

    async def _handle_search_code(self, text: str) -> str:
        m = re.search(r"(?:search|find)\s+(?:code|file)\s+[\"'](.+?)[\"']", text, re.I)
        if not m:
            m = re.search(r"(?:search|find)\s+(?:code|file)\s+(\S+)", text, re.I)
        if not m:
            return "What to search? Usage: `search code 'function_name'`"
        return await self.github.search_code(m.group(1))

    async def _handle_search_repos(self, text: str) -> str:
        m = re.search(r"(?:search|find)\s+(?:repo|repos)\s+(\S+)", text, re.I)
        if not m:
            return "What to search? Usage: `search repos python`"
        return await self.github.search_repos(m.group(1))

    async def _handle_list_issues(self, text: str) -> str:
        m = re.search(r"(?:issue|issues|bug|bugs)\s+(?:list|show)\s+(\S+)", text, re.I)
        if not m:
            return "Which repo? Usage: `issues list my-project`"
        return await self.github.list_issues(m.group(1))

    async def _handle_create_issue(self, text: str) -> str:
        m = re.search(
            r"(?:create|new|open)\s+(?:issue|bug)\s+(?:in|for)\s+(\S+)\s+[\"'](.+?)[\"']",
            text, re.I,
        )
        if not m:
            m = re.search(
                r"(?:create|new|open)\s+(?:issue|bug)\s+[\"'](.+?)[\"']\s+(?:in|for)\s+(\S+)",
                text, re.I,
            )
            if m:
                return await self.github.create_issue(m.group(2), m.group(1))
            return "Usage: `create issue in my-project 'Bug title'`"
        return await self.github.create_issue(m.group(1), m.group(2))

    # ── Vikunja handlers ────────────────────────────────────────

    async def _handle_list_projects(self) -> str:
        try:
            records = self.projects.discover()
        except Exception:
            return "Could not reach Vikunja. Is it running?"
        if not records:
            return "No active projects in Vikunja."
        lines = [f"**Vikunja Projects ({len(records)}):**"]
        for r in records:
            s = r.signals
            status = []
            if s.open_tasks:
                status.append(f"{s.open_tasks} open")
            if s.done_tasks:
                status.append(f"{s.done_tasks} done")
            if s.overdue_tasks:
                status.append(f"{s.overdue_tasks} overdue!")
            status_str = ", ".join(status) if status else "no tasks"
            lines.append(f"  **{r.name}** — {status_str}")
            if r.description:
                lines.append(f"    {r.description[:80]}")
        return "\n".join(lines)

    async def _handle_list_tasks(self, text: str) -> str:
        # Try to extract project name
        m = re.search(r"(?:task|tasks)\s+(?:list|show)\s+(?:in|for|of)\s+(\S+)", text, re.I)
        if not m:
            return "Which project? Usage: `tasks list project-name`"
        project_name = m.group(1)
        try:
            records = self.projects.discover()
        except Exception:
            return "Could not reach Vikunja."
        for r in records:
            if r.name.lower() == project_name.lower():
                titles = self.projects.list_task_titles(r.vikunja_id)
                if not titles:
                    return f"No tasks in **{r.name}**."
                lines = [f"**{r.name}** tasks ({len(titles)}):"]
                for t in titles[:15]:
                    lines.append(f"  - {t}")
                if len(titles) > 15:
                    lines.append(f"  ... and {len(titles) - 15} more")
                return "\n".join(lines)
        return f"Project '{project_name}' not found."

    async def _handle_create_task(self, text: str) -> str:
        m = re.search(
            r"(?:create|new|add)\s+task\s+[\"'](.+?)[\"']\s+(?:in|for|to)\s+(\S+)",
            text, re.I,
        )
        if not m:
            m = re.search(
                r"(?:create|new|add)\s+task\s+(?:in|for|to)\s+(\S+)\s+[\"'](.+?)[\"']",
                text, re.I,
            )
            if m:
                project_name, title = m.group(1), m.group(2)
            else:
                return "Usage: `create task 'My task' in project-name`"
        else:
            title, project_name = m.group(1), m.group(2)

        try:
            records = self.projects.discover()
        except Exception:
            return "Could not reach Vikunja."
        for r in records:
            if r.name.lower() == project_name.lower():
                result = self.projects.create_task(r.vikunja_id, title)
                if result:
                    self.memory.add(
                        f"[fact] Created Vikunja task '{title}' in '{project_name}'",
                        metadata={"type": "extracted_fact", "action": "create_task"},
                    )
                    return f"Created task **{title}** in **{project_name}**."
                return f"Failed to create task in {project_name}."
        return f"Project '{project_name}' not found."

    async def _handle_close_task(self, text: str) -> str:
        return "Task closing not yet implemented. Do it in Vikunja for now."

    # ── Session handlers ────────────────────────────────────────

    async def _handle_start_session(self, text: str, user_id: int) -> str:
        m = re.search(r"(?:for|about|called|named)\s+[\"']?(.+?)[\"']?\s*$", text, re.I)
        title = m.group(1) if m else f"discord-{user_id}"
        try:
            result = await self.opencode.create_session(title=title)
            session_id = result.get("id", result.get("sessionID", ""))
            self._dm_sessions[user_id] = session_id
            self.memory.add(
                f"[fact] Started opencode session '{title}' (id: {session_id})",
                metadata={"type": "session_start", "session_id": session_id},
            )
            return f"Started session **{title}** (`{session_id}`). Send coding questions and I'll use this session."
        except Exception:
            return "Could not start session. Is opencode serve running?"

    async def _handle_list_sessions(self) -> str:
        try:
            sessions = await self.opencode.list_sessions()
        except Exception:
            return "Could not reach opencode serve."
        if not sessions:
            return "No active sessions."
        lines = [f"**Sessions ({len(sessions)}):**"]
        for s in sessions[:10]:
            title = s.get("title", "untitled")
            sid = s.get("id", s.get("sessionID", ""))[:12]
            lines.append(f"  **{title}** — `{sid}...`")
        return "\n".join(lines)

    async def _handle_session_status(self) -> str:
        try:
            statuses = await self.opencode.session_status()
        except Exception:
            return "Could not reach opencode serve."
        if not statuses:
            return "No session status available."
        lines = ["**Session Status:**"]
        for sid, info in statuses.items():
            if isinstance(info, dict):
                status = info.get("status", info.get("type", "unknown"))
            else:
                status = str(info)
            lines.append(f"  `{sid[:12]}...` — {status}")
        return "\n".join(lines)

    # ── Memory handlers ─────────────────────────────────────────

    async def _handle_memory_search(self, user_id: int, text: str = "") -> str:
        # Extract search term or just show recent memories
        m = re.search(r"(?:search|find|recall|remember)\s+(.+)", text, re.I) if text else None
        query = m.group(1) if m else "session summary"
        results = self.memory.search(query, top_k=5)
        if not results:
            return f"No memories found for '{query}'."
        lines = [f"**Memories for '{query}':**"]
        for r in results:
            mem = r.get("memory", "")[:120]
            lines.append(f"  - {mem}")
        return "\n".join(lines)

    async def _handle_briefing(self) -> str:
        lines = ["**Secretary Briefing**", ""]

        # Recent memories
        recent = self.memory.get_all(limit=5)
        if recent:
            lines.append("**Recent activity:**")
            for m in recent[:3]:
                lines.append(f"  - {m.get('memory', '')[:80]}")
            lines.append("")

        # Vikunja status
        try:
            records = self.projects.discover()
            if records:
                lines.append(f"**Vikunja:** {len(records)} active projects")
                for r in records[:3]:
                    s = r.signals
                    if s.open_tasks:
                        lines.append(f"  - {r.name}: {s.open_tasks} open tasks")
        except Exception:
            lines.append("**Vikunja:** unavailable")

        # GitHub
        try:
            code, out, _ = await self.github._run(
                f"gh repo list {self.github.default_owner} --limit 5 --json name"
            )
            if code == 0:
                import json
                repos = json.loads(out) if out else []
                lines.append(f"\n**GitHub:** {len(repos)} recent repos shown")
        except Exception:
            pass

        # OpenCode
        try:
            sessions = await self.opencode.list_sessions()
            lines.append(f"**OpenCode:** {len(sessions)} active sessions")
        except Exception:
            lines.append("**OpenCode:** unavailable")

        return "\n".join(lines)

    # ── Default: opencode forward ───────────────────────────────

    async def _handle_opencode(self, text: str, user_id: int) -> str:
        """Forward to opencode serve for coding assistance."""
        session_id = self._dm_sessions.get(user_id)
        if not session_id:
            try:
                result = await self.opencode.create_session(
                    title=f"discord-dm-{user_id}"
                )
                session_id = result.get("id", result.get("sessionID", ""))
                self._dm_sessions[user_id] = session_id
            except Exception:
                return (
                    "I can't reach opencode serve right now. "
                    "I can still help with GitHub repos, Vikunja projects, and memory."
                )

        try:
            response = await self.opencode.send_message(session_id, text)
            reply = self.opencode.extract_text(response)
            self.memory.add(
                f"[discord:{user_id}] Q: {text[:200]} A: {reply[:200]}",
                metadata={"type": "discord_exchange", "user_id": str(user_id)},
            )
            return reply
        except Exception:
            return "Failed to get response from opencode. Try again."

    # ── Helpers ─────────────────────────────────────────────────

    async def _send_long(self, channel: Any, text: str) -> None:
        """Send a message, splitting if >2000 chars."""
        if len(text) <= 2000:
            await channel.send(text)
        else:
            for i in range(0, len(text), 2000):
                await channel.send(text[i : i + 2000])

    def _help_text(self) -> str:
        return """**What I can do:**

**GitHub:**
  `create repo my-project` — new GitHub repo
  `list repos` — show all your repos
  `repo info my-project` — repo details
  `delete repo my-project` — delete a repo
  `clone my-project` — clone locally
  `search code 'query'` — search code
  `search repos python` — search repos
  `issues list my-project` — list issues
  `create issue in my-project 'title'` — create issue

**Vikunja Projects:**
  `list projects` — show all projects + task counts
  `tasks list project-name` — list tasks in a project
  `create task 'Task title' in project-name` — create task

**Sessions:**
  `start session for my-feature` — new opencode session
  `list sessions` — show active sessions

**Memory:**
  `briefing` — get a status summary
  `search <query>` — search memories
  `what did we work on` — recent activity

**Or just talk to me** — I'll forward coding questions to opencode."""

    # ── Slash commands ──────────────────────────────────────────

    @app_commands.command(name="status", description="Show secretary status")
    async def status_command(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        await interaction.followup.send(await self._handle_briefing(), ephemeral=True)

    @app_commands.command(name="briefing", description="Get a status briefing")
    async def briefing_command(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        await interaction.followup.send(await self._handle_briefing(), ephemeral=True)

    @app_commands.command(name="search", description="Search memories")
    @app_commands.describe(query="What to search for")
    async def search_command(
        self, interaction: discord.Interaction, query: str
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        results = self.memory.search(query, top_k=5)
        if not results:
            await interaction.followup.send("No memories found.", ephemeral=True)
            return
        lines = [f"**Search: {query}**"]
        for r in results:
            lines.append(f"- {r.get('memory', '')[:100]}")
        await interaction.followup.send("\n".join(lines), ephemeral=True)


def _matches(text: str, pattern: str) -> bool:
    """Check if text matches a regex pattern (case-insensitive)."""
    return bool(re.search(pattern, text, re.I))


async def run_bot(config: dict[str, Any], opencode: OpenCodeClient, memory: MemoryStore) -> None:
    """Run the Discord bot (blocks until disconnected)."""
    discord_cfg = config.get("discord", {})
    token = discord_cfg.get("token", "")
    if not token:
        logger.error("No Discord bot token configured (discord.token in config.yaml)")
        return

    bot = SecretaryBot(config, opencode, memory)
    try:
        await bot.start(token)
    except discord.LoginFailure:
        logger.error("Invalid Discord bot token")
    except Exception:
        logger.exception("Discord bot crashed")
    finally:
        await bot.close()
