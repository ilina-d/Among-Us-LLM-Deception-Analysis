"""Per-agent knowledge graph used as the agent's memory (``--memory kg``).

Replaces the free-text ``[Condensed Memory]`` that the model writes for itself. The graph is
built deterministically from what the agent has *actually perceived* (no extra LLM calls, no
ground truth it could not have seen) and is rendered into the prompt's ``Memory:`` slot.

Graph schema (networkx MultiDiGraph)
------------------------------------
Nodes   player:<name>   attrs: kind="player", dead (bool), dead_room, dead_t
        room:<name>     attrs: kind="room"
        task:<name>     attrs: kind="task"
        utt:<n>         attrs: kind="utterance", text, speaker, meeting, round, t
Edges   player -[IN]->      room   t, src ("self" | "seen")          where a player was
        player -[MOVED]->   room   t, frm, vent(bool), src            observed movement / vent
        player -[TASK]->    room   t, src, done(bool), fake(bool)     seen doing a task / own task
        player -[KILLED]->  player t, room, src                       observed (or own) kill
        player -[REPORTED]->room   t, meeting_called(bool)
        player -[VOTED]->   player t, meeting
        player -[SAID]->    utt    t, meeting
        player -[SAW_BODY]->player t, room                            (a body was in my room)
Every edge also has ``src`` so that first-hand facts stay distinguishable from statements.
Statements are stored verbatim and attributed; the graph never turns a statement into a fact.

The agent sees everything since the start of the game, whereas the baseline prompt only shows the
last 4 observations and 4 actions plus the model's own summary.
"""
from __future__ import annotations

import re
from typing import Optional

import networkx as nx

_OBS_RE = re.compile(
    r"^Timestep (?P<t>\d+): \[(?P<phase>task|meeting phase - round (?P<rnd>\d+))\] "
    r"(?P<who>Player \d+: \w+) (?P<what>.*)$", re.S)
_TIME_RE = re.compile(r"Game Time: (\d+)/(\d+)")
_LOC_RE = re.compile(r"Current Location: (.+)")
_PLAYERS_RE = re.compile(r"Players in [^:\n]+: (.*)")
_MOVE_RE = re.compile(r"^(MOVE|VENT) from (.+?) to (.+)$")
_KILL_RE = re.compile(r"^KILL (Player \d+: \w+)")
_VOTE_RE = re.compile(r"^VOTE (Player \d+: \w+)")
_REPORT_RE = re.compile(r"^REPORT DEAD BODY at (.+)$")
_BUTTON_RE = re.compile(r"^CALL MEETING using the emergency button at (.+)$")
_TASK_RE = re.compile(r"^COMPLETE (FAKE )?TASK - (.+)$")


class KnowledgeGraph:
    def __init__(self, owner: str, max_statement_chars: int = 220, max_events_per_player: int = 10):
        self.owner = owner
        self.g = nx.MultiDiGraph()
        self.max_chars = max_statement_chars
        self.max_events = max_events_per_player
        self.t = 0
        self.phase = "task"
        self.meeting_no = 0
        self.room = None
        self._obs_seen = 0
        self._act_seen = 0
        self._utt = 0
        self._pnode(owner)

    # ---- node helpers -------------------------------------------------------------------
    def _pnode(self, name: str) -> str:
        n = f"player:{name}"
        if n not in self.g:
            self.g.add_node(n, kind="player", name=name, dead=False)
        return n

    def _rnode(self, name: str) -> str:
        n = f"room:{name}"
        if n not in self.g:
            self.g.add_node(n, kind="room", name=name)
        return n

    def _tnode(self, name: str) -> str:
        n = f"task:{name}"
        if n not in self.g:
            self.g.add_node(n, kind="task", name=name)
        return n

    def _add_in(self, u: str, v: str, **attrs) -> None:
        """IN edges are re-derived from the location text every turn; keep one per (player, room, t, src)."""
        for d in (self.g.get_edge_data(u, v) or {}).values():
            if d["rel"] == "IN" and d["t"] == self.t and d["src"] == attrs["src"]:
                return
        self.g.add_edge(u, v, rel="IN", t=self.t, **attrs)

    def _say(self, speaker: str, text: str, t: int, rnd: Optional[int]):
        self._utt += 1
        u = f"utt:{self._utt}"
        self.g.add_node(u, kind="utterance", text=text.strip(), speaker=speaker, meeting=self.meeting_no,
                        round=rnd, t=t)
        self.g.add_edge(self._pnode(speaker), u, rel="SAID", t=t, meeting=self.meeting_no)

    # ---- ingestion ----------------------------------------------------------------------
    def update(self, player) -> None:
        """Pull everything new out of the player object. Call once per decision, before prompting."""
        self._ingest_location(player.location_info or "", player.location)
        obs = player.observation_history
        for m in obs[self._obs_seen:]:
            self._ingest_observation(m)
        self._obs_seen = len(obs)
        acts = player.action_history
        for rec in acts[self._act_seen:]:
            self._ingest_own_action(rec)
        self._act_seen = len(acts)

    def _ingest_location(self, text: str, location: Optional[str]) -> None:
        m = _TIME_RE.search(text)
        if m:
            self.t = int(m.group(1))
        in_meeting = "Current phase: Meeting" in text
        if in_meeting and self.phase != "meeting":
            self.meeting_no += 1
        self.phase = "meeting" if in_meeting else "task"
        m = _LOC_RE.search(text)
        room = (m.group(1).strip() if m else None) or location
        if not room:
            return
        self.room = room
        self._add_in(self._pnode(self.owner), self._rnode(room), src="self",
                     meeting=self.meeting_no if in_meeting else None)
        pm = _PLAYERS_RE.search(text)
        if not pm:
            return
        for raw in pm.group(1).split(", "):
            raw = raw.strip()
            if not raw:
                continue
            dead = raw.endswith("(dead)")
            name = raw.replace("(dead)", "").strip()
            if not name or name == self.owner:
                continue
            pn = self._pnode(name)
            if dead:
                if not self.g.nodes[pn]["dead"]:
                    self.g.nodes[pn].update(dead=True, dead_room=room, dead_t=self.t)
                    self.g.add_edge(self._pnode(self.owner), pn, rel="SAW_BODY", t=self.t, room=room)
            elif not in_meeting:
                self._add_in(pn, self._rnode(room), src="seen")

    def _ingest_observation(self, msg: str) -> None:
        m = _OBS_RE.match(msg.strip())
        if not m:
            return
        t, who, what = int(m.group("t")), m.group("who"), m.group("what").strip()
        rnd = int(m.group("rnd")) if m.group("rnd") else None
        pn = self._pnode(who)
        here = self.room or "?"
        if what.startswith("SPEAK:"):
            self._say(who, what[len("SPEAK:"):], t, rnd)
        elif _MOVE_RE.match(what):
            kind, frm, to = _MOVE_RE.match(what).groups()
            self.g.add_edge(pn, self._rnode(to), rel="MOVED", t=t, frm=frm, vent=(kind == "VENT"), src="seen")
        elif _KILL_RE.match(what):
            victim = _KILL_RE.match(what).group(1)
            self.g.add_edge(pn, self._pnode(victim), rel="KILLED", t=t, room=here, src="seen")
        elif _VOTE_RE.match(what):
            self.g.add_edge(pn, self._pnode(_VOTE_RE.match(what).group(1)), rel="VOTED", t=t,
                            meeting=self.meeting_no)
        elif _REPORT_RE.match(what):
            self.g.add_edge(pn, self._rnode(_REPORT_RE.match(what).group(1)), rel="REPORTED", t=t,
                            meeting_called=False)
        elif _BUTTON_RE.match(what):
            self.g.add_edge(pn, self._rnode(_BUTTON_RE.match(what).group(1)), rel="REPORTED", t=t,
                            meeting_called=True)
        elif what.startswith("Seemingly doing task"):
            self.g.add_edge(pn, self._rnode(here), rel="TASK", t=t, src="seen", done=False, fake=None)

    def _ingest_own_action(self, rec: dict) -> None:
        act = rec["action"]
        t = rec["timestep"]
        text = str(act)
        me = self._pnode(self.owner)
        if text.startswith("SPEAK:"):
            self._say(self.owner, text[len("SPEAK:"):], t, rec.get("round"))
        elif _MOVE_RE.match(text):
            kind, frm, to = _MOVE_RE.match(text).groups()
            self.g.add_edge(me, self._rnode(to), rel="MOVED", t=t, frm=frm, vent=(kind == "VENT"), src="self")
        elif _KILL_RE.match(text):
            self.g.add_edge(me, self._pnode(_KILL_RE.match(text).group(1)), rel="KILLED", t=t,
                            room=self.room or "?", src="self")
        elif _VOTE_RE.match(text):
            self.g.add_edge(me, self._pnode(_VOTE_RE.match(text).group(1)), rel="VOTED", t=t,
                            meeting=self.meeting_no)
        elif _TASK_RE.match(text):
            fake, name = _TASK_RE.match(text).groups()
            self.g.add_edge(me, self._tnode(name), rel="TASK", t=t, src="self", done=True, fake=bool(fake),
                            room=self.room or "?")
        elif _REPORT_RE.match(text):
            self.g.add_edge(me, self._rnode(_REPORT_RE.match(text).group(1)), rel="REPORTED", t=t,
                            meeting_called=False)
        elif _BUTTON_RE.match(text):
            self.g.add_edge(me, self._rnode(_BUTTON_RE.match(text).group(1)), rel="REPORTED", t=t,
                            meeting_called=True)

    # ---- queries ------------------------------------------------------------------------
    def _out(self, node: str, rel: str):
        return sorted(((v, d) for _, v, d in self.g.out_edges(node, data=True) if d["rel"] == rel),
                      key=lambda x: x[1]["t"])

    def my_path(self) -> list[tuple[int, str]]:
        steps, last = [], None
        for v, d in self._out(self._pnode(self.owner), "IN"):
            room = self.g.nodes[v]["name"]
            label = f"{room} [meeting {d['meeting']}]" if d.get("meeting") else room
            if label != last:
                steps.append((d["t"], label))
                last = label
        return steps

    def sightings(self, name: str) -> list[str]:
        """First-hand sightings/observations of one player, oldest to newest, as short strings."""
        pn = self._pnode(name)
        ev = []
        for v, d in self._out(pn, "IN"):
            ev.append((d["t"], 0, f"t{d['t']}: in {self.g.nodes[v]['name']}"))
        for v, d in self._out(pn, "MOVED"):
            how = "VENTED" if d["vent"] else "moved"
            ev.append((d["t"], 1, f"t{d['t']}: {how} {d['frm']}->{self.g.nodes[v]['name']}"))
        for v, d in self._out(pn, "TASK"):
            if d.get("src") == "seen":
                ev.append((d["t"], 1, f"t{d['t']}: seemed to do a task"))
        for v, d in self._out(pn, "KILLED"):
            ev.append((d["t"], 2, f"t{d['t']}: KILLED {self.g.nodes[v]['name']} in {d['room']}"))
        for v, d in self._out(pn, "REPORTED"):
            what = "called a meeting" if d["meeting_called"] else "reported a body"
            ev.append((d["t"], 2, f"t{d['t']}: {what} at {self.g.nodes[v]['name']}"))
        ev.sort(key=lambda x: (x[0], x[1]))
        # collapse repeated identical "in <room>" runs
        out, last_in = [], None
        for _, k, s in ev:
            if k == 0:
                room = s.split("in ", 1)[1]
                if room == last_in:
                    continue
                last_in = room
            else:
                last_in = None
            out.append(s)
        return out[-self.max_events:]

    def meetings(self) -> dict[int, list[tuple[str, str, int]]]:
        out: dict[int, list] = {}
        for n, d in self.g.nodes(data=True):
            if d.get("kind") == "utterance":
                out.setdefault(d["meeting"], []).append((d["t"], d["speaker"], d["text"], n))
        return {k: sorted(v, key=lambda x: (x[0], int(x[3].split(":")[1]))) for k, v in sorted(out.items())}

    def votes(self) -> dict[int, list[str]]:
        out: dict[int, list] = {}
        for u, v, d in self.g.edges(data=True):
            if d["rel"] == "VOTED":
                out.setdefault(d["meeting"], []).append(
                    f"{self.g.nodes[u]['name']} -> {self.g.nodes[v]['name']}")
        return dict(sorted(out.items()))

    # ---- rendering ----------------------------------------------------------------------
    def render(self) -> str:
        me = self._pnode(self.owner)
        L = ["KNOWLEDGE GRAPH (your memory, rebuilt every turn from everything you have perceived since the game "
             "began; 'seen' = first-hand observation, statements are only what a player SAID, not verified facts)."]
        path = self.my_path()
        L.append("You were in: " + (" -> ".join(f"{r}(t{t})" for t, r in path) if path else "(no record yet)"))
        tasks = [(d["t"], self.g.nodes[v]["name"], d.get("room", "?"), d["fake"])
                 for v, d in self._out(me, "TASK") if d.get("done")]
        L.append("Tasks you completed: " + ("; ".join(
            f"{n} in {r} (t{t}{', fake' if f else ''})" for t, n, r, f in tasks) if tasks else "none yet"))
        kills = [(d["t"], self.g.nodes[v]["name"], d["room"]) for v, d in self._out(me, "KILLED")]
        if kills:
            L.append("Your kills: " + "; ".join(f"{n} in {r} (t{t})" for t, n, r in kills))
        others = sorted(n for n, d in self.g.nodes(data=True)
                        if d.get("kind") == "player" and d["name"] != self.owner)
        L.append("Players you have observed:")
        if not others:
            L.append("  (nobody yet)")
        for n in others:
            d = self.g.nodes[n]
            status = f" [DEAD, body first seen in {d['dead_room']} at t{d['dead_t']}]" if d["dead"] else ""
            ev = self.sightings(d["name"])
            L.append(f"  {d['name']}{status}: " + ("; ".join(ev) if ev else "no first-hand observations"))
        ms = self.meetings()
        votes = self.votes()
        for k in sorted(set(ms) | set(votes)):
            L.append(f"Meeting {k}:")
            for t, speaker, text, _ in ms.get(k, []):
                text = " ".join(text.split())
                if len(text) > self.max_chars:
                    text = text[: self.max_chars - 3] + "..."
                L.append(f"  {speaker} said: \"{text}\"")
            if votes.get(k):
                L.append("  Votes: " + "; ".join(votes[k]))
        return "\n".join(L)

    def stats(self) -> dict:
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}
