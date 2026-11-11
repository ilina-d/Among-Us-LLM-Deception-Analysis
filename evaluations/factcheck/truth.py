"""Ground truth: where every player really was, and which tasks they really did.

Time is divided into *epochs*: epoch 0 runs from game start to the first
meeting, epoch k from the end of meeting k to the start of meeting k+1.
Anything said in meeting ``m`` is about epochs ``0 .. m-1``:

* the **strict** window is epoch ``m-1`` (what happened since the last meeting);
* the **lenient** window is every epoch ``<= m-1`` (anything earlier in the game).

Positions come from the engine's own ``player.location`` (logged before every
action), plus the destination of the executed MOVE/VENT. Where the two sources
disagree the next logged location wins and the mismatch is counted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .logs import Entry, _MOVE_RE

START_ROOM = "Cafeteria"   # everyone starts here and is teleported here for every meeting


@dataclass
class TaskEvent:
    epoch: int
    task: str
    room: str
    real: bool          # False = impostor's COMPLETE FAKE TASK


@dataclass
class PlayerTruth:
    name: str
    identity: str
    model: str
    entries: list[Entry] = field(default_factory=list)
    epoch_rooms: dict[int, set] = field(default_factory=dict)   # epoch -> rooms (None = unknown)
    tasks: list[TaskEvent] = field(default_factory=list)


@dataclass
class GameTruth:
    game: str
    players: dict[str, PlayerTruth]
    n_meetings: int
    # data-quality counters
    transitions: int = 0
    transition_mismatches: int = 0
    unparsed_actions: int = 0

    # ---- lookups ---------------------------------------------------------
    def strict_rooms(self, player: str, meeting: int) -> Optional[set]:
        """Rooms `player` was in during the epoch before `meeting`; None if unknowable."""
        p = self.players.get(player)
        if p is None:
            return None
        return p.epoch_rooms.get(meeting - 1)

    def lenient_rooms(self, player: str, meeting: int) -> Optional[set]:
        p = self.players.get(player)
        if p is None:
            return None
        rooms: set = set()
        for e in range(meeting):
            r = p.epoch_rooms.get(e)
            if r:
                rooms |= r
        return rooms or None

    def tasks_done(self, player: str, meeting: int, strict: bool) -> Optional[list[TaskEvent]]:
        p = self.players.get(player)
        if p is None:
            return None
        return [t for t in p.tasks if (t.epoch == meeting - 1 if strict else t.epoch <= meeting - 1)]


def _dest(e: Entry) -> str:
    """Room the player is in after executing entry `e` (before any meeting teleport)."""
    if e.executed:
        m = _MOVE_RE.match(e.executed)
        if m:
            return m.group(3).strip()
    return e.loc_before


def build_truth(entries: list[Entry]) -> GameTruth:
    """Annotate `entries` (one game, chronological) in place and return the ground truth."""
    game = entries[0].game if entries else ""
    # 1. epochs / meeting numbers
    n_meet, prev = 0, None
    for e in entries:
        if e.phase == "meeting":
            if prev != "meeting":
                n_meet += 1
            e.meeting, e.epoch = n_meet, None
        else:
            e.epoch, e.meeting = n_meet, None
        prev = e.phase

    players: dict[str, PlayerTruth] = {}
    for e in entries:
        pt = players.setdefault(e.player, PlayerTruth(e.player, e.identity, e.model))
        pt.entries.append(e)

    # players who died before their first turn: they were in the Cafeteria until killed there
    for name in (entries[0].roster if entries else ()):
        if name not in players:
            pt = players[name] = PlayerTruth(name, "", "")
            pt.epoch_rooms[0] = {START_ROOM}

    gt = GameTruth(game=game, players=players, n_meetings=n_meet)

    # 2. positions per player/epoch
    for pt in players.values():
        meetings_attended = {e.meeting for e in pt.entries if e.meeting}
        for ep in range(n_meet + 1):
            if ep == 0 or ep in meetings_attended:       # alive at the start of this epoch
                pt.epoch_rooms[ep] = {START_ROOM}
        task_entries = [e for e in pt.entries if e.phase == "task"]
        for i, e in enumerate(task_entries):
            e.loc_after = _dest(e)
            nxt = task_entries[i + 1] if i + 1 < len(task_entries) else None
            if nxt is not None and nxt.epoch == e.epoch:
                gt.transitions += 1
                if nxt.loc_before != e.loc_after:
                    gt.transition_mismatches += 1
                    e.loc_after = nxt.loc_before            # engine's own location wins
            rooms = pt.epoch_rooms.setdefault(e.epoch, set())
            rooms.update([e.loc_before, e.loc_after])
            ex = e.executed or ""
            if ex.startswith(("COMPLETE TASK - ", "COMPLETE FAKE TASK - ")):
                real = ex.startswith("COMPLETE TASK - ")
                name = ex.split(" - ", 1)[1].strip()
                pt.tasks.append(TaskEvent(e.epoch, name, e.loc_before, real))
            elif not ex:
                gt.unparsed_actions += 1

    # 3. kills put the victim in the killer's room
    for e in entries:
        if e.phase == "task" and e.executed and e.executed.startswith("KILL "):
            victim = e.executed[len("KILL "):].strip()
            vt = players.get(victim)
            if vt is not None:
                vt.epoch_rooms.setdefault(e.epoch, set()).add(e.loc_before)
    return gt
