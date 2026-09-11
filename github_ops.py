"""GitHub operations — wraps `gh` CLI for repo/issue management.

All operations go through the authenticated `gh` CLI (already logged in
as 0124212). No API keys needed — uses the system gh auth.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger("secretary.github")


async def _run(cmd: str) -> tuple[int, str, str]:
    """Run a shell command async, return (exitcode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode().strip(), stderr.decode().strip()


class GitHubOps:
    """Async wrapper around gh CLI for repo/issue/project operations."""

    def __init__(self, default_owner: str = "0124212") -> None:
        self.default_owner = default_owner

    # ── Repos ───────────────────────────────────────────────────

    async def list_repos(self, limit: int = 20, public_only: bool = False) -> str:
        """List repos for the default owner."""
        visibility = "--visibility public" if public_only else ""
        code, out, err = await _run(
            f"gh repo list {self.default_owner} --limit {limit} --json "
            f"name,description,isPrivate,updatedAt,url {visibility}"
        )
        if code != 0:
            return f"Error listing repos: {err}"
        repos = json.loads(out) if out else []
        if not repos:
            return "No repos found."
        lines = []
        for r in repos:
            vis = "private" if r.get("isPrivate") else "public"
            desc = r.get("description", "") or ""
            lines.append(f"**{r['name']}** [{vis}] — {desc[:60]}")
            lines.append(f"  {r['url']}")
        return "\n".join(lines)

    async def create_repo(
        self,
        name: str,
        description: str = "",
        private: bool = False,
        auto_init: bool = True,
        gitignore: str = "",
        license: str = "",
    ) -> str:
        """Create a new GitHub repo."""
        vis = "--private" if private else "--public"
        cmd = f"gh repo create {self.default_owner}/{name} {vis}"
        if description:
            cmd += f' --description "{description}"'
        if auto_init:
            cmd += " --clone"
        if gitignore:
            cmd += f" --gitignore {gitignore}"
        if license:
            cmd += f" --license {license}"

        code, out, err = await _run(cmd)
        if code != 0:
            return f"Error creating repo: {err}"
        return f"Created repo: {out}" if out else "Repo created (no output)."

    async def delete_repo(self, name: str) -> str:
        """Delete a repo (requires confirmation in output)."""
        code, out, err = await _run(
            f"gh repo delete {self.default_owner}/{name} --yes"
        )
        if code != 0:
            return f"Error deleting repo: {err}"
        return f"Deleted {self.default_owner}/{name}."

    async def get_repo(self, name: str) -> str:
        """Get detailed info about a repo."""
        code, out, err = await _run(
            f"gh repo view {self.default_owner}/{name} --json "
            f"name,description,isPrivate,createdAt,updatedAt,url,"
            f"defaultBranchRef,languages,repositoryTopics,"
            f"forkCount,starCount,watchers,issues,pullRequests"
        )
        if code != 0:
            return f"Error: {err}"
        r = json.loads(out) if out else {}
        if not r:
            return f"Repo {name} not found."
        vis = "private" if r.get("isPrivate") else "public"
        lines = [
            f"**{r['name']}** [{vis}]",
            f"Description: {r.get('description', 'none')}",
            f"URL: {r.get('url', 'n/a')}",
            f"Created: {r.get('createdAt', 'n/a')}",
            f"Updated: {r.get('updatedAt', 'n/a')}",
            f"Stars: {r.get('starCount', 0)} | Forks: {r.get('forkCount', 0)}",
            f"Open issues: {len(r.get('issues', []))} | PRs: {len(r.get('pullRequests', []))}",
        ]
        langs = r.get("languages", [])
        if langs:
            lines.append(f"Languages: {', '.join(l.get('name', '') for l in langs)}")
        topics = r.get("repositoryTopics", [])
        if topics:
            lines.append(f"Topics: {', '.join(t.get('topic', {}).get('name', '') for t in topics)}")
        return "\n".join(lines)

    # ── Issues ──────────────────────────────────────────────────

    async def list_issues(
        self, repo: str, state: str = "open", limit: int = 10
    ) -> str:
        """List issues in a repo."""
        code, out, err = await _run(
            f"gh issue list --repo {self.default_owner}/{repo} "
            f"--state {state} --limit {limit} --json number,title,state,labels,createdAt"
        )
        if code != 0:
            return f"Error: {err}"
        issues = json.loads(out) if out else []
        if not issues:
            return f"No {state} issues in {repo}."
        lines = [f"**{repo}** — {len(issues)} {state} issue(s):"]
        for i in issues:
            labels = ", ".join(l.get("name", "") for l in i.get("labels", []))
            label_str = f" [{labels}]" if labels else ""
            lines.append(f"  #{i['number']} {i['title']}{label_str}")
        return "\n".join(lines)

    async def create_issue(
        self,
        repo: str,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
    ) -> str:
        """Create an issue in a repo."""
        cmd = f"gh issue create --repo {self.default_owner}/{repo} --title '{title}'"
        if body:
            cmd += f" --body '{body}'"
        if labels:
            cmd += f" --label {','.join(labels)}"
        code, out, err = await _run(cmd)
        if code != 0:
            return f"Error: {err}"
        return f"Created issue: {out}"

    # ── Searches ────────────────────────────────────────────────

    async def search_code(self, query: str, repo: str = "", limit: int = 5) -> str:
        """Search code across repos."""
        search = f"{query} owner:{self.default_owner}"
        if repo:
            search += f" repo:{self.default_owner}/{repo}"
        code, out, err = await _run(
            f"gh search code '{search}' --limit {limit} --json path,repository,textMatches"
        )
        if code != 0:
            return f"Error: {err}"
        results = json.loads(out) if out else []
        if not results:
            return "No code matches found."
        lines = [f"**Code search: {query}**"]
        for r in results:
            path = r.get("path", "")
            repo_name = r.get("repository", {}).get("name", "")
            lines.append(f"  {repo_name}/{path}")
        return "\n".join(lines)

    async def search_repos(self, query: str, limit: int = 5) -> str:
        """Search repos by keyword."""
        code, out, err = await _run(
            f"gh search repos '{query} owner:{self.default_owner}' --limit {limit} "
            f"--json name,description,url,isPrivate"
        )
        if code != 0:
            return f"Error: {err}"
        repos = json.loads(out) if out else []
        if not repos:
            return "No matching repos."
        lines = [f"**Repo search: {query}**"]
        for r in repos:
            vis = "private" if r.get("isPrivate") else "public"
            desc = (r.get("description") or "")[:50]
            lines.append(f"  **{r['name']}** [{vis}] — {desc}")
        return "\n".join(lines)

    # ── Clones ──────────────────────────────────────────────────

    async def clone(self, repo: str, dest: str = "") -> str:
        """Clone a repo to a local path."""
        url = f"https://github.com/{self.default_owner}/{repo}.git"
        target = dest or f"/root/repos/{repo}"
        code, out, err = await _run(f"git clone {url} {target} 2>&1")
        if code != 0:
            return f"Clone failed: {err or out}"
        return f"Cloned to {target}"

    async def list_local_repos(self) -> str:
        """List repos cloned in /root/repos/."""
        import os
        repos_dir = "/root/repos"
        if not os.path.isdir(repos_dir):
            return "No /root/repos/ directory."
        entries = sorted(os.listdir(repos_dir))
        if not entries:
            return "No repos in /root/repos/."
        lines = ["**Local repos:**"]
        for e in entries:
            if os.path.isdir(os.path.join(repos_dir, e)):
                lines.append(f"  {e}")
        return "\n".join(lines)
