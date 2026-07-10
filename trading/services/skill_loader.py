# -*- coding: utf-8 -*-
"""Agent Skills loader — versioned domain playbooks for the chat agent.

The handoff's modernization note paired MCP with the Agent Skills pattern:
"MCP gives agents hands, Skills give them judgment." A skill is a folder
under ``skills/`` containing a ``SKILL.md`` with a small YAML-ish
frontmatter block (name, description, triggers) followed by markdown
instructions. Instead of hardcoding "how to interpret an RSI divergence"
or "how to judge optimizer output" into prompt strings scattered across
the codebase, the chat pipeline loads the relevant playbook on demand,
keyed off the user's message.

Wiring: pages/6_Chat.py passes the rendered block through
``execute_with_tools(platform_context_suffix=...)`` — the parameter that
exists for exactly this purpose — so matched skills ride alongside the
memory/agent context on every turn that needs them.

Format (SKILL.md):

    ---
    name: signal-interpretation
    description: How to read Evolve's technical signals without fooling yourself
    triggers: rsi, macd, bollinger, divergence, overbought, oversold, signal
    ---
    # markdown body ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Keep the injected block bounded so skills can't crowd out the
# conversation itself.
MAX_SKILLS_PER_TURN = 2
MAX_SKILL_CHARS = 4000


@dataclass
class Skill:
    name: str
    description: str
    triggers: List[str]
    body: str
    path: str = ""

    def matches(self, message: str) -> int:
        """Number of trigger phrases present in the message (0 = no match)."""
        msg = (message or "").lower()
        return sum(1 for t in self.triggers if t and t in msg)


def _parse_skill_md(text: str, path: str = "") -> Optional[Skill]:
    """Parse frontmatter + body. Returns None if the frontmatter is absent
    or missing required fields (a malformed skill should be skipped loudly
    in logs, never crash the chat turn)."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        logger.warning("Skill %s missing frontmatter; skipped", path)
        return None
    try:
        _, front, body = stripped.split("---", 2)
    except ValueError:
        logger.warning("Skill %s has unterminated frontmatter; skipped", path)
        return None
    meta: Dict[str, str] = {}
    for line in front.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    name = meta.get("name", "")
    if not name:
        logger.warning("Skill %s missing 'name'; skipped", path)
        return None
    triggers = [t.strip().lower() for t in meta.get("triggers", "").split(",") if t.strip()]
    return Skill(
        name=name,
        description=meta.get("description", ""),
        triggers=triggers,
        body=body.strip()[:MAX_SKILL_CHARS],
        path=path,
    )


def _default_skills_dir() -> Path:
    # repo_root/skills — this file lives at repo_root/trading/services/
    return Path(__file__).resolve().parents[2] / "skills"


_cache: Dict[str, List[Skill]] = {}


def discover_skills(skills_dir: Optional[Path] = None, use_cache: bool = True) -> List[Skill]:
    """Find every skills/<name>/SKILL.md and parse it."""
    root = Path(skills_dir) if skills_dir else _default_skills_dir()
    key = str(root)
    if use_cache and key in _cache:
        return _cache[key]
    skills: List[Skill] = []
    if root.is_dir():
        for md in sorted(root.glob("*/SKILL.md")):
            try:
                skill = _parse_skill_md(md.read_text(encoding="utf-8"), str(md))
                if skill:
                    skills.append(skill)
            except Exception as e:  # noqa: BLE001 - one bad skill never kills chat
                logger.warning("Skill load failed for %s: %s", md, e)
    _cache[key] = skills
    return skills


def match_skills(message: str, skills_dir: Optional[Path] = None) -> List[Skill]:
    """Skills relevant to a message, best matches first, capped."""
    scored = [
        (s.matches(message), s) for s in discover_skills(skills_dir)
    ]
    hits = sorted(
        (pair for pair in scored if pair[0] > 0),
        key=lambda p: p[0],
        reverse=True,
    )
    return [s for _, s in hits[:MAX_SKILLS_PER_TURN]]


def render_skills_context(message: str, skills_dir: Optional[Path] = None) -> str:
    """The context suffix for a chat turn: matched playbooks, or ''.

    Never raises — a skills problem must never break a chat turn.
    """
    try:
        matched = match_skills(message, skills_dir)
        if not matched:
            return ""
        parts = ["\n\n## Loaded skill playbooks (follow these when relevant)"]
        for s in matched:
            parts.append(f"\n### Skill: {s.name}\n{s.body}")
        return "\n".join(parts)
    except Exception as e:  # noqa: BLE001
        logger.warning("Skill context rendering failed: %s", e)
        return ""
