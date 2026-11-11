"""Load ``agent-logs-compact.json`` files into a flat list of per-turn entries.

The agent logs record, for every turn of every player: the player's *real*
location (``player.location``, taken from the engine just before the action),
the full prompt (which lists the available actions) and the raw model response.
The engine does not log which action it actually executed, so we replay its
(deterministic) response-to-action matching to recover it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HEAD = "[Condensed Memory]"
_TP = "[Thinking Process]"
_ACT = "[Action]"
_MOVE_RE = re.compile(r"^(MOVE|VENT) from (.+?) to (.+)$")


@dataclass
class Entry:
    game: str                 # unique across experiments: "<experiment>/Game 3"
    seq: int                  # chronological position within the game
    step: int                 # engine timestep (all of a meeting shares one)
    timestamp: str
    player: str               # "Player 3: blue"
    identity: str             # "Crewmate" | "Impostor"
    model: str
    phase: str                # "task" | "meeting"
    loc_before: str           # engine location before acting
    output_action: str        # text the engine matched actions against
    executed: Optional[str]   # replayed engine decision (repr of the action)
    format_ok: bool           # response followed [Condensed Memory]/[Thinking Process]/[Action]
    api_fallback: bool        # response is the hard-coded 'SPEAK: ...' failure fallback
    fell_through: bool = False     # [Action] matched no legal action; engine silently ran the last one
    timestamp_raw_game: str = ""   # game_index as written in the log ("Game 3")
    roster: set = field(default_factory=set, repr=False)  # every "Player N: colour" seen in this game's prompts
    # filled in by truth.build_truth
    epoch: Optional[int] = None    # task entries: number of meetings started before
    meeting: Optional[int] = None  # meeting entries: 1-based meeting number
    loc_after: Optional[str] = None

    @property
    def speech(self) -> Optional[str]:
        """The message other players actually heard (None if not a SPEAK)."""
        if self.executed and self.executed.startswith("SPEAK: "):
            return self.executed[len("SPEAK: "):]
        return None

    @property
    def number(self) -> Optional[int]:
        m = re.match(r"Player\s*(\d+)", self.player)
        return int(m.group(1)) if m else None

    @property
    def color(self) -> str:
        return self.player.split(":", 1)[1].strip().lower() if ":" in self.player else ""


def engine_output_action(resp: str) -> tuple[str, bool]:
    """Port of the regex in ``LLMAgent.choose_action``.

    Returns (text after the last [Action], whether the format regex matched);
    if it did not match the engine uses the whole stripped response.
    """
    if resp.startswith(_HEAD):
        a = resp.rfind(_ACT)
        if a != -1 and resp.rfind(_TP, len(_HEAD), a) != -1:
            return resp[a + len(_ACT):].strip(), True
    return resp.strip(), False


def parse_available_actions(prompt) -> list[str]:
    text = prompt.get("All Info", "") if isinstance(prompt, dict) else str(prompt or "")
    i = text.rfind("Available actions:")
    if i == -1:
        return []
    out = []
    for line in text[i:].splitlines()[1:]:
        m = re.match(r"^\s*\d+\.\s(.*)$", line)
        if m:
            out.append(m.group(1).rstrip())
    return out


def emulate_executed(available: list[str], output_action: str) -> Optional[str]:
    """Replay the engine's matching loop (agent.py: for action in available_actions ...)."""
    if not available:
        return None
    for a in available:
        if a in output_action:
            return a
        if "SPEAK: " in a and "SPEAK: " in output_action:
            return "SPEAK: " + output_action.split("SPEAK: ")[1]
    return available[-1]  # engine falls through to the last available action


def fell_through(available: list[str], output_action: str) -> bool:
    """True if the model's [Action] matched none of the legal actions, so the engine ran the last one."""
    if not available:
        return False
    return not any(a in output_action or ("SPEAK: " in a and "SPEAK: " in output_action) for a in available)


def _structured_action_text(resp) -> str:
    """Fallback for old logs without ``full_response``: dig the chosen action out of the parsed response.

    Older log versions nest it under different section names ("Action", "ACTION",
    "Thinking Process" -> "action", "SPEAK Strategy" -> "action", ...).
    """
    if isinstance(resp, str):
        return resp.strip()
    if isinstance(resp, dict):
        for k, v in resp.items():                       # an explicit action key first, at any depth
            if str(k).lower().strip("[] ") == "action" or str(k).lower().endswith(" action"):
                t = _structured_action_text(v)
                if t:
                    return t
        for v in resp.values():
            if isinstance(v, dict):
                t = _structured_action_text(v)
                if t:
                    return t
    return ""


def _infer_phase(prompt, output_action: str, executed: Optional[str]) -> str:
    if isinstance(prompt, dict) and prompt.get("Phase"):
        return "meeting" if "meeting" in str(prompt["Phase"]).lower() else "task"
    text = prompt.get("All Info", "") if isinstance(prompt, dict) else str(prompt or "")
    if "Current phase: Meeting" in text or "Meeting phase" in text[:400]:
        return "meeting"
    act = executed or output_action
    return "meeting" if act.startswith(("SPEAK", "VOTE")) else "task"


_NAME_RE = re.compile(r"Player\s+\d+:\s+[A-Za-z]+")


def parse_entry(raw: dict, game: str, seq: int) -> Entry:
    inter = raw.get("interaction", {}) or {}
    prompt = inter.get("prompt")
    full = inter.get("full_response")
    if not isinstance(full, str):
        full = _structured_action_text(inter.get("response"))
    out, fmt_ok = engine_output_action(full)
    avail = parse_available_actions(prompt)
    executed = emulate_executed(avail, out)
    pl = raw.get("player", {}) or {}
    return Entry(
        game=game, seq=seq, step=int(raw.get("step", -1)), timestamp=str(raw.get("timestamp", "")),
        player=pl.get("name", ""), identity=pl.get("identity", ""), model=str(pl.get("model", "")),
        phase=_infer_phase(prompt, out, executed), loc_before=pl.get("location", ""),
        output_action=out, executed=executed, format_ok=fmt_ok, fell_through=fell_through(avail, out),
        api_fallback=full.strip() == "SPEAK: ...", timestamp_raw_game=str(raw.get("game_index", "")),
    )


def load_log_file(path: str | Path, experiment: str = "") -> dict[str, list[Entry]]:
    """Read one agent-logs-compact.json (JSONL). Returns {game_key: entries in order}.

    Games run concurrently so lines of different games interleave, but order
    *within* a game is chronological, which is all we rely on.
    """
    games: dict[str, list[Entry]] = {}
    rosters: dict[str, set] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line of an interrupted run
            gid = str(raw.get("game_index", "Game ?"))
            key = f"{experiment}/{gid}" if experiment else gid
            lst = games.setdefault(key, [])
            entry = parse_entry(raw, key, len(lst))
            # players who die before their first turn never get a log line of their own, but
            # they appear in other players' prompts and can still be talked about in meetings
            prompt = (raw.get("interaction", {}) or {}).get("prompt")
            text = prompt.get("All Info", "") if isinstance(prompt, dict) else str(prompt or "")
            names = rosters.setdefault(key, set())
            names.update(re.sub(r"\s+", " ", n) for n in _NAME_RE.findall(text))
            names.add(entry.player)
            entry.roster = names
            lst.append(entry)
    return games


def load_experiment(logs_root: str | Path, experiment: str) -> dict[str, list[Entry]]:
    return load_log_file(Path(logs_root) / experiment / "agent-logs-compact.json", experiment)
