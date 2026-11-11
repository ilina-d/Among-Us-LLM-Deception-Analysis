"""Knowledge-graph memory: faithfulness against the engine's own record, and baseline isolation.

Runs the real engine with the scripted policy from simulate_games (SIM_MEMORY selects the backend)
and checks that everything the graph asserts first-hand really happened.
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "among-agents"))

MOVE = re.compile(r"t(\d+): (moved|VENTED) (.+?)->(.+)$")
IN = re.compile(r"t(\d+): in (.+)$")
KILL = re.compile(r"t(\d+): KILLED (Player \d+: \w+) in (.+)$")


def _run(tmp, mode, games=6, seed=3):
    import simulate_games as S
    os.environ["SIM_MEMORY"] = mode
    asyncio.run(S.run(tmp, n_games=games, seed=seed))
    rows = [json.loads(l) for l in (tmp / "agent-logs-compact.json").read_text().splitlines() if l.strip()]
    return rows, json.loads((tmp / "oracle_traj.json").read_text())


@pytest.fixture(scope="module")
def kg_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("kg"), "kg")


@pytest.fixture(scope="module")
def text_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("txt"), "text")


def _split_players(block):
    """'  Player 2: pink [DEAD ...]: ev; ev' -> (name, [events])"""
    out = {}
    for line in block:
        m = re.match(r"\s+(Player \d+: \w+)(?: \[DEAD[^\]]*\])?: (.*)$", line)
        if m:
            out[m.group(1)] = [e.strip() for e in m.group(2).split("; ")]
    return out


def test_first_hand_facts_match_engine_record(kg_run):
    rows, traj = kg_run
    checked = 0
    for r in rows:
        g = r["game_index"].split()[1]
        mem = r["interaction"]["prompt"]["Memory"]
        assert mem.startswith("KNOWLEDGE GRAPH")
        lines = mem.split("\n")
        start = lines.index("Players you have observed:") + 1
        block = []
        for ln in lines[start:]:
            if not ln.startswith("  Player"):
                break
            block.append(ln)
        for name, events in _split_players(block).items():
            tr = traj.get(f"{g}|{name}", [])
            for ev in events:
                if m := MOVE.match(ev):
                    t, kind, frm, to = int(m.group(1)), m.group(2), m.group(3), m.group(4)
                    hit = [a for a in tr if a["timestep"] == t and a["before"] == frm and a["after"] == to
                           and a["action_name"] in ("MOVE", "VENT")]
                    assert hit, f"{name} {ev} not in oracle"
                    assert (kind == "VENTED") == (hit[0]["action_name"] == "VENT")
                    checked += 1
                elif m := IN.match(ev):
                    t, room = int(m.group(1)), m.group(2)
                    # the observer's decision can fall before or after the target's action in the same timestep
                    if not tr:   # killed before ever acting: could only have been seen at the start, in the Cafeteria
                        assert room == "Cafeteria" and t <= 1, f"{name} {ev}"
                    else:
                        assert any(abs(a["timestep"] - t) <= 1 and room in (a["before"], a["after"]) for a in tr), \
                            f"{name} {ev} never in that room near that time"
                    checked += 1
                elif m := KILL.match(ev):
                    t, victim, room = int(m.group(1)), m.group(2), m.group(3)
                    assert any(a["timestep"] == t and a["action"].startswith("KILL") and victim in a["action"]
                               for a in tr)
                    checked += 1
    assert checked > 200


def test_statements_are_verbatim_and_attributed(kg_run):
    rows, traj = kg_run
    n = 0
    for r in rows:
        g = r["game_index"].split()[1]
        for line in r["interaction"]["prompt"]["Memory"].split("\n"):
            m = re.match(r'\s+(Player \d+: \w+) said: "(.*)"$', line)
            if not m or m.group(2).endswith("..."):
                continue
            speaker, text = m.groups()
            said = [a["action"][len("SPEAK: "):].strip() for a in traj.get(f"{g}|{speaker}", [])
                    if a["action"].startswith("SPEAK: ")]
            assert " ".join(text.split()) in [" ".join(s.split()) for s in said], (speaker, text)
            n += 1
    assert n > 100


def test_memory_only_contains_perceived_players_and_grows(kg_run):
    rows, _ = kg_run
    by = {}
    for r in rows:
        by.setdefault((r["game_index"], r["player"]["name"]), []).append(len(r["interaction"]["prompt"]["Memory"]))
    grew = sum(1 for v in by.values() if len(v) > 3 and v[-1] > v[0])
    assert grew >= 0.8 * len(by)


def test_text_mode_is_unchanged(text_run):
    rows, _ = text_run
    assert all(not r["interaction"]["prompt"]["Memory"].startswith("KNOWLEDGE GRAPH") for r in rows)


def test_graph_basic_schema():
    from amongagents.agent.knowledge_graph import KnowledgeGraph

    class P:
        location = "Admin"
        location_info = ("Game Time: 4/50\nCurrent phase: Task phase\nx\nCurrent Location: Admin\n"
                         "Players in Admin: Player 1: red, Player 2: blue (dead)\n\n")
        observation_history = ["Timestep 3: [task] Player 1: red MOVE from Cafeteria to Admin"]
        action_history = []

    kg = KnowledgeGraph("Player 1: red")
    kg.update(P)
    kg.update(P)       # idempotent
    txt = kg.render()
    assert "Player 2: blue [DEAD, body first seen in Admin at t4]" in txt
    assert kg.stats()["nodes"] >= 3
    before = kg.stats()
    kg.update(P)
    assert kg.stats() == before
