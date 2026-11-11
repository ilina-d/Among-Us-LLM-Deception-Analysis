"""Extract checkable factual claims from a player's speech.

Deliberately rule-based: free, deterministic, and every extracted claim keeps
the sentence and the rule that produced it so precision can be audited.
The extractor favours precision over recall - it only emits a claim when the
sentence clearly asserts *where someone was* or *that the speaker finished a
task*. Questions, hearsay ("X said ..."), quotes and hedges are either skipped
or flagged (``hedged`` / ``hearsay``) and left out of the headline statistics.

Claim kinds
-----------
self_location   "I was in Admin", "I came from Storage", "I wasn't in Medbay"
other_location  "I saw Player 3 in Electrical", "Blue was in O2"
self_task       "I finished Fix Wiring"  (checked against real vs fake task completions)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

ROOM_ALIASES = {
    "Cafeteria": ["cafeteria", "cafe"],
    "Weapons": ["weapons"],
    "Navigation": ["navigation", "nav"],
    "O2": ["o2"],
    "Shields": ["shields"],
    "Communications": ["communications", "communication", "comms"],
    "Storage": ["storage"],
    "Admin": ["admin"],
    "Electrical": ["electrical", "elec"],
    "Lower Engine": ["lower engine", "lower-engine"],
    "Security": ["security"],
    "Reactor": ["reactor"],
    "Upper Engine": ["upper engine", "upper-engine"],
    "Medbay": ["medbay", "med bay", "med-bay", "medical bay", "medical"],
}
COLORS = ["red", "blue", "green", "pink", "orange", "yellow", "black", "white",
          "purple", "brown", "cyan", "lime"]

# canonical task -> regexes (canonical spelling is added automatically); order = match priority
TASK_ALIASES = {
    "Accept Diverted Power": [r"accept(?:ed|ing)?\s+(?:the\s+)?diverted\s+power"],
    "Divert Power": [r"divert(?:ed|ing)?\s+(?:the\s+)?power"],
    "Align Engine Output": [r"align(?:ed|ing)?\s+(?:the\s+)?engine(?:\s+output)?"],
    "Calibrate Distributor": [r"calibrat(?:e|ed|ing)\s+(?:the\s+)?distributor"],
    "Chart Course": [r"chart(?:ed|ing)?\s+(?:the\s+|a\s+)?course"],
    "Clean O2 Filter": [r"clean(?:ed|ing)?\s+(?:the\s+)?o2(?:\s+filter)?", r"o2\s+filter"],
    "Clear Asteroids": [r"clear(?:ed|ing)?\s+(?:the\s+)?asteroids"],
    "Download Data": [r"download(?:ed|ing)?\s+(?:the\s+)?data"],
    "Upload Data": [r"upload(?:ed|ing)?\s+(?:the\s+)?data"],
    "Empty Chute": [r"empt(?:y|ied|ying)\s+(?:the\s+)?chute"],
    "Empty Garbage": [r"empt(?:y|ied|ying)\s+(?:the\s+)?garbage"],
    "Fix Wiring": [r"fix(?:ed|ing)?\s+(?:the\s+|my\s+)?wir(?:ing|es)", r"wir(?:ing|es)"],
    "Fuel Engines": [r"fuel(?:ed|ing)?\s+(?:the\s+)?engines?"],
    "Inspect Sample": [r"inspect(?:ed|ing)?\s+(?:the\s+)?sample"],
    "Prime Shields": [r"prim(?:e|ed|ing)\s+(?:the\s+)?shields"],
    "Stabilize Steering": [r"stabili[sz](?:e|ed|ing)\s+(?:the\s+)?steering"],
    "Start Reactor": [r"start(?:ed|ing)?\s+(?:the\s+)?reactor"],
    "Submit Scan": [r"submit(?:ted|ting)?\s+(?:the\s+|a\s+|my\s+)?scan"],
    "Swipe Card": [r"swip(?:e|ed|ing)\s+(?:the\s+|my\s+|a\s+)?card", r"card\s+swipe"],
    "Unlock Manifolds": [r"unlock(?:ed|ing)?\s+(?:the\s+)?manifolds"],
}


def _sp(s: str) -> str:
    return re.escape(s).replace(r"\ ", r"[ -]").replace(r"\-", r"[ -]")


_ROOM_ALT = "|".join(sorted((_sp(a) for al in ROOM_ALIASES.values() for a in al), key=len, reverse=True))
_ROOM_LOOKUP = {a.replace("-", " "): c for c, al in ROOM_ALIASES.items() for a in al}
_COLOR_ALT = "|".join(COLORS)

ROOM = rf"(?:the\s+)?(?P<room>{_ROOM_ALT})\b(?:\s+room)?"
# a reference to a player: "Player 3", "Player 3: blue", "Player 3 (blue)", or a bare colour
REF = rf"(?:Player\s*\d+(?:\s*:\s*(?:{_COLOR_ALT})\b|\s*\(\s*(?:{_COLOR_ALT})\s*\))?|\b(?:{_COLOR_ALT})\b)"
REFS = rf"(?P<chain>{REF}(?:\s*(?:,|&|and)\s*{REF})*)"
REF_FIND = re.compile(rf"Player\s*(?P<num>\d+)(?:\s*[:(]\s*(?P<col>{_COLOR_ALT}))?|\b(?P<col2>{_COLOR_ALT})\b", re.I)

I = r"(?-i:I)"
ADV = r"(?:(?:just|also|only|still|briefly|quickly|already|then|first|initially|mostly|actually|really|simply|currently|even|there|here|later|finally|next|immediately|had|have)\s+)*"
OBS = (r"saw|seen|noticed|spotted|observed|watched|met|encountered|found|witnessed|ran\s+into|bumped\s+into|"
       r"passed|see|notice|spot")
# words that signal the subject of the clause changed - location scanning stops there
CUT_RE = re.compile(rf"\b(?:{OBS}|Player\s*\d+|you|they|he|she|we|them|him|her|me|no\s*one|nobody|everyone|anyone|someone|"
                    rf"the\s+body|body|{_COLOR_ALT})\b|(?-i:\bI\b)", re.I)
_PREP = (r"to|into|through|from|via|in|at|inside|passing|passed|left|exited|entered|leaving|entering|exiting|"
         r"enter|leave|exit")
_PREP_ROOM = re.compile(rf"\b(?:{_PREP})\s+{ROOM}", re.I)
_INAT_ROOM = re.compile(rf"\b(?:in|at|inside)\s+{ROOM}", re.I)
_LISTCONT = re.compile(rf"\s*(?:,\s*(?:and\s+)?|\band\b\s*(?:then\s+)?|\bthen\b\s*|&\s*|->\s*|→\s*)(?:(?:to|in|at|into|through)\s+)?{ROOM}", re.I)
_TASK_NAMES = list(TASK_ALIASES)
_TASK_RE = re.compile("|".join(
    f"(?P<t{i}>{'|'.join([_sp(n.lower())] + TASK_ALIASES[n])})" for i, n in enumerate(_TASK_NAMES)), re.I)

HEDGE_RE = re.compile(r"\b(?:if|maybe|perhaps|might|may|could|would|should|probably|possibly|think|thought|believe|"
                      r"believed|guess|suppose|assume|assumed|suspect|wonder|unsure|not\s+sure|seems?|seemed|"
                      r"apparently|supposedly|allegedly|hypothetically|imagine|let's|lets|hope|hoping|pretend|"
                      r"whether|unless|wish)\b", re.I)
HEARSAY_RE = re.compile(r"\b(?:said|says|claim(?:s|ed)|told|mention(?:s|ed)|stated|states|reported|admitted|insisted|"
                        r"argued|noted|asserted|alleged|swore|whispered|announced|confirmed)\b|\baccording\s+to\b", re.I)
_DELIM_RE = re.compile(r"[,;:]|\b(?:and|but|so|because|since)\b", re.I)

_MOTION = (r"went|gone|moved|walked|headed|ran|came|entered|arrived|passed|left|vented|travell?ed|returned|stopped|got|"
           r"ended\s+up|swung|exited|stepped|proceeded|made\s+my\s+way")
_DONE = (r"completed|finished|did|done|accomplished|handled|performed|fixed|swiped|uploaded|downloaded|emptied|"
         r"cleaned|fueled|fuelled|calibrated|aligned|diverted|submitted|primed|charted|stabili[sz]ed|unlocked|"
         r"started|cleared|accepted|inspected")
_NOT_GERUND = (r"heading|going|planning|trying|about|looking|hoping|thinking|wanting|gonna|supposed|aiming|intending|"
               r"waiting\s+for|searching|asking|telling|saying|suspecting|accusing|voting|calling|following|"
               r"watching|wondering|guessing|checking\s+on|worried")
_BE_NOW = r"(?:was|were|am|is)"
_EXCL_RE = re.compile(r"\b(?:the\s+)?(?:whole|entire)\s+(?:time|round|meeting|duration)\b|\ball\s+along\b|"
                      r"\bnever\s+left\b|\bthroughout\b", re.I)
# "... was in Admin when Player 5 was killed and then moved to Electrical": same subject, new room
_CONT_RE = re.compile(rf"\band\s+(?:then\s+|later\s+)?(?:\w+ly\s+)*(?:moved|went|walked|ran|headed|vented|rushed|"
                      rf"hurried|came)\s+(?:\w+ly\s+)?(?:to|into|through)\s+{ROOM}", re.I)


@dataclass
class Claim:
    kind: str                       # self_location | other_location | self_task
    subject: Optional[str]          # full player name, or None if unresolvable
    room: Optional[str] = None
    task: Optional[str] = None
    negated: bool = False
    exclusive: bool = False         # "I stayed in X" / "I was in X the whole time": never left X
    hedged: bool = False
    hearsay: bool = False
    sentence: str = ""
    rule: str = ""

    def key(self):
        return (self.kind, self.subject, self.room, self.task, self.negated, self.exclusive)


@dataclass
class Roster:
    by_number: dict = field(default_factory=dict)
    by_color: dict = field(default_factory=dict)

    @classmethod
    def from_names(cls, names):
        r = cls()
        for n in names:
            m = re.match(r"Player\s*(\d+)\s*:\s*(\w+)", n)
            if m:
                r.by_number[int(m.group(1))] = n
                r.by_color[m.group(2).lower()] = n
        return r

    def resolve(self, ref: str) -> Optional[str]:
        m = REF_FIND.search(ref)
        if not m:
            return None
        if m.group("num") is not None:
            return self.by_number.get(int(m.group("num")))
        return self.by_color.get((m.group("col2") or "").lower())


def _canon_room(s: str) -> str:
    return _ROOM_LOOKUP[re.sub(r"[ -]+", " ", s.lower()).strip()]


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[*_`#]+", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


_QUESTION_RE = re.compile(
    r"^(?:(?:additionally|also|and|but|so|then|now|finally|lastly|however|first|second|next),?\s+)*"
    r"(?:has|have|did|does|do|can|could|would|will|should|is|are|was|were|why|what|where|when|who|how)\b"
    r"|\b(?:can|could|would|will)\s+(?:you|anyone|anybody|someone|everyone|player\s*\d+)\b.*\b(?:confirm|explain|clarify|tell|share|say)\b"
    r"|\bhas\s+anyone\b|\bhave\s+(?:you|any\s+of\s+you)\b|\bdid\s+(?:anyone|you|anybody)\b", re.I)


def is_question(sent: str) -> bool:
    """A question that lost its '?' (models often end a final question without one)."""
    return sent.rstrip().endswith("?") or bool(_QUESTION_RE.search(sent.strip().strip('"')))


def split_clauses(text: str) -> list[str]:
    """Sentences, with the trailing question clause removed from questions."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+|;\s*", text):
        s = s.strip()
        if not s:
            continue
        if s.endswith("?"):
            parts = s.split(",")
            s = ",".join(parts[:-1]).strip()   # keep any statement before the question
            if not s:
                continue
        out.append(s)
    return out


def _rooms_after(sent: str, start: int, rx=_PREP_ROOM, max_gap: int = 70, first_only: bool = False,
                 skip: int = 0) -> list[str]:
    """Rooms named (with a location preposition) after `start`, up to the next change of subject."""
    tail = sent[start:]
    cut = CUT_RE.search(tail, skip)
    if cut:
        tail = tail[:cut.start()]
    rooms, pos = [], 0
    while True:
        m = rx.search(tail, pos)
        if not m or (not rooms and m.start() > max_gap):
            break
        rooms.append(_canon_room(m.group("room")))
        if first_only:
            break
        pos = _extend_list(tail, m.end(), rooms)
    return rooms


def _extend_list(text: str, pos: int, rooms: list) -> int:
    """Append "..., Storage and then Electrical"-style continuations to `rooms`."""
    while True:
        lc = _LISTCONT.match(text, pos)
        if not lc:
            return pos
        rooms.append(_canon_room(lc.group("room")))
        pos = lc.end()


def _context_flags(sent: str, start: int) -> tuple[bool, bool]:
    """(hedged, hearsay) judged from the words just before the claim in the same clause."""
    pre = sent[:start]
    d = None
    for d in _DELIM_RE.finditer(pre):
        pass
    pre_clause = pre[d.end():] if d else pre
    hedged = bool(HEDGE_RE.search(pre_clause[-45:]))
    hearsay = False
    window = pre[-70:]
    for m in HEARSAY_RE.finditer(window):
        before = window[:m.start()].rstrip().split()
        if not before or before[-1] not in ("I", "I've", "I'd"):
            hearsay = True
    if pre.count('"') % 2 == 1:
        hearsay = True
    return hedged, hearsay


def extract_claims(message: str, speaker: str, roster: Roster) -> list[Claim]:
    """All claims in `message` (a SPEAK message by `speaker`), de-duplicated per sentence."""
    claims: list[Claim] = []
    for sent in split_clauses(normalize(message)):
        if is_question(sent):
            continue
        found: dict = {}

        def add(kind, who, rooms, rule, start, task=None, negated=False, exclusive=False):
            for subj in ([who] if isinstance(who, str) else who):
                if subj is None:
                    continue
                k = "self_location" if (kind == "other_location" and subj == speaker) else kind
                for r in rooms:
                    if r is None and k != "self_task":
                        continue
                    c = Claim(k, subj, r, task, negated, exclusive, sentence=sent, rule=rule)
                    c.hedged, c.hearsay = _context_flags(sent, start)
                    found.setdefault(c.key(), c)

        def refs(chain):
            return [roster.resolve(m.group(0)) for m in REF_FIND.finditer(chain)]

        def after_room(m, cont=False):
            rooms = [_canon_room(m.group("room"))]
            pos = _extend_list(sent, m.end(), rooms)
            if cont:
                cm = _CONT_RE.search(sent, pos)
                if cm:
                    rooms.append(_canon_room(cm.group("room")))
            return rooms

        # self, negated: "I wasn't in Admin", "I never went to Admin"
        for m in re.finditer(
                rf"{I}\s+(?:(?:was|am)\s+not|wasn't|weren't|'m\s+not|never\s+(?:was|been|went|go|entered|visited|moved|"
                rf"walked|headed|came|stepped)|haven't\s+been|have\s+not\s+been|hadn't\s+been|(?:didn't|did\s+not)\s+"
                rf"(?:go|visit|enter|move|walk|head|come|step))(?:\s+(?:even|really|at\s+all))*\s*"
                rf"(?:(?:in|at|inside|to|into|through)\s+)?{ROOM}", sent, re.I):
            add("self_location", speaker, [_canon_room(m.group("room"))], "self_neg", m.start(), negated=True)
        # self: "I was in X" / "I'm at X"
        for m in re.finditer(rf"{I}(?:\s+(?:was|am|have\s+been|had\s+been)|'m|'ve\s+been|'d\s+been)\s+{ADV}"
                             rf"(?:in|at|inside)\s+{ROOM}", sent, re.I):
            excl = bool(_EXCL_RE.search(sent[m.end():m.end() + 70]))
            add("self_location", speaker, after_room(m, cont=not excl), "self_be", m.start(), exclusive=excl)
        # self: "I stayed in X" (never left)
        for m in re.finditer(rf"{I}\s+(?:(?:just|only|simply|also)\s+)*(?:stayed|remained)\s+(?:in|at|inside)\s+{ROOM}",
                             sent, re.I):
            add("self_location", speaker, [_canon_room(m.group("room"))], "self_stayed", m.start(), exclusive=True)
        # self: "I was doing/fixing ... in X"
        for m in re.finditer(rf"{I}(?:\s+{_BE_NOW}|'m)\s+(?!(?:{_NOT_GERUND})\b)\w+ing\b", sent, re.I):
            add("self_location", speaker, _rooms_after(sent, m.end(), _INAT_ROOM, 60, True), "self_ing", m.start())
        # self, motion: "I went to X", "I came from X", "I walked from X to Y", "I left X"
        for m in re.finditer(rf"{I}\s+{ADV}(?P<verb>{_MOTION})\b", sent, re.I):
            add("self_location", speaker,
                _rooms_after(sent, m.start("verb"), skip=len(m.group("verb"))), "self_motion", m.start())
        # self: "I checked/visited X"
        for m in re.finditer(rf"{I}\s+{ADV}(?:checked|visited|cleared|searched|scouted|swept|explored|"
                             rf"went\s+through)\s+(?:out\s+)?{ROOM}", sent, re.I):
            add("self_location", speaker, after_room(m), "self_checked", m.start())
        # self: "I finished <task> in X"  -> a task claim, plus a location claim
        for m in re.finditer(rf"{I}\s+{ADV}(?:successfully\s+)?(?P<v>{_DONE})\b|"
                             rf"{I}(?:'m|\s+am)\s+(?:done|finished)\s+with\b", sent, re.I):
            v = m.group("v") or ""
            start = m.start("v") if v else m.end()
            tail = sent[start:]
            cut = CUT_RE.search(tail, len(v))
            if cut:
                tail = tail[:cut.start()]
            tm = _TASK_RE.search(tail)
            rooms = _rooms_after(sent, start, _INAT_ROOM, 80, True, skip=len(v))
            if tm:
                task = _TASK_NAMES[int(tm.lastgroup[1:])]
                add("self_task", speaker, [rooms[0] if rooms else None], "self_task", m.start(), task=task)
            if rooms and (tm or re.search(r"\btasks?\b", tail, re.I)):
                add("self_location", speaker, rooms, "self_task_room", m.start())
        # self + others: "I was with Player 3 in X"
        for m in re.finditer(rf"{I}\s+(?:was|were|am|'m)\s+(?:(?:just|also|only|there|together|actually)\s+)*"
                             rf"(?:with|alongside)\s+(?P<who>[^.;!?]{{1,60}}?)\s+(?:together\s+)?(?:in|at|inside)\s+{ROOM}",
                             sent, re.I):
            r = after_room(m)
            add("self_location", speaker, r, "self_with", m.start())
            add("other_location", [w for w in refs(m.group("who")) if w], r, "self_with", m.start())
        # self + others: "Player 3 and I were in X" / "I and Player 3 were in X"
        for pat in (rf"{REFS}\s+and\s+{I}", rf"{I}\s+and\s+{REFS}"):
            for m in re.finditer(pat + rf"\s+(?:were|was)\s+(?:(?:both|together|also|just)\s+)*(?:in|at|inside)\s+{ROOM}",
                                 sent, re.I):
                r = after_room(m)
                add("self_location", speaker, r, "we_were", m.start())
                add("other_location", [w for w in refs(m.group("chain")) if w], r, "we_were", m.start())
        # others: "Player 3 was in X"
        for m in re.finditer(rf"{REFS}\s+(?:was|were|is|had\s+been|has\s+been)\s+"
                             rf"(?:(?:just|also|only|still|there|seen|standing|alone|actually|already|definitely|clearly|sitting)\s+)*"
                             rf"(?:in|at|inside)\s+{ROOM}", sent, re.I):
            add("other_location", [w for w in refs(m.group("chain")) if w], after_room(m, cont=True), "other_be", m.start())
        for m in re.finditer(rf"{REFS}\s+{_BE_NOW}\s+(?!(?:{_NOT_GERUND})\b)\w+ing\b", sent, re.I):
            add("other_location", [w for w in refs(m.group("chain")) if w],
                _rooms_after(sent, m.end(), _INAT_ROOM, 50, True), "other_ing", m.start())
        # others, motion: "Player 3 went to X", "Blue left Admin"
        for m in re.finditer(rf"{REFS}\s+{ADV}(?P<verb>{_MOTION})\b", sent, re.I):
            add("other_location", [w for w in refs(m.group("chain")) if w],
                _rooms_after(sent, m.start("verb"), skip=len(m.group("verb"))), "other_motion", m.start())
        # others, observed: "I saw Player 3 in X", "I noticed Blue enter Admin"
        for m in re.finditer(rf"{I}\s+(?:(?:also|just|briefly|quickly|clearly|definitely|actually|did|then|first|last|only|even|"
                             rf"have|had|'ve)\s+)*(?:{OBS})\s+(?:that\s+)?{REFS}", sent, re.I):
            add("other_location", [w for w in refs(m.group("chain")) if w], _rooms_after(sent, m.end()),
                "other_seen", m.start())
        # others, bare: "Player 3 in Admin"
        for m in re.finditer(rf"{REFS}\s*,?\s*(?:in|at)\s+{ROOM}", sent, re.I):
            add("other_location", [w for w in refs(m.group("chain")) if w], after_room(m), "other_bare", m.start())
        claims.extend(found.values())
    return claims
