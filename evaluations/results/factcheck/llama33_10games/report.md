# Fact-check report: llama33_10games

## What was measured

Every `SPEAK` message was scanned for claims about **where a player was** and **which tasks the speaker finished**. Each claim was checked against the game record rebuilt from the agent logs (the engine's own `player.location` before every action, plus replayed move/task/kill actions). A claim is a *false claim* when the record contradicts it; this counts honest mistakes (misremembering) as well as deliberate lies, so the crewmate rate is best read as a floor on memory/hallucination errors and the impostor rate as errors plus deception.

Two readings of 'when': the headline (**conservative**) label calls a positive claim false only if the player was *never* there earlier in the game, and a negated claim ('I wasn't in X') false only if they were there since the last meeting. The **strict** label requires positive claims to hold since the last meeting.

## Coverage

|  | n |
|---|---|
| Games | 10 |
| Speeches | 195 |
| Speeches with >=1 checkable claim | 60 |
| Claims extracted | 141 |
| ...hedged ('I think...') / hearsay ('X said...') / Cafeteria-only (cannot be false) | 3 / 17 / 22 |
| ...repeats of the same claim in the same meeting | 48 |
| Claims counted in headline rates | 63 |

## Who lies more: impostors or crewmates?

| Unit | Role | n | false | False-claim rate [95% CI] |
|---|---|---|---|---|
| claims | Crewmate | 46 | 1 | 2.2% [0.0%, 7.5%] |
| claims | Impostor | 17 | 5 | 29.4% [19.0%, 46.2%] |
| claims | **Impostor - Crewmate** |  |  | **+27.2 pp** [+18.8, +41.0], p=0.00025 |
| speeches (>=1 false claim) | Crewmate | 42 | 1 | 2.4% [0.0%, 7.5%] |
| speeches (>=1 false claim) | Impostor | 18 | 7 | 38.9% [23.1%, 64.3%] |
| speeches (>=1 false claim) | **Impostor - Crewmate** |  |  | **+36.5 pp** [+21.1, +59.7], p=0.00025 |
| player-games (>=1 false claim) | Crewmate | 21 | 1 | 4.8% [0.0%, 15.0%] |
| player-games (>=1 false claim) | Impostor | 9 | 5 | 55.6% [33.3%, 85.7%] |
| player-games (>=1 false claim) | **Impostor - Crewmate** |  |  | **+50.8 pp** [+28.0, +75.0], p=0.00025 |

_CIs and p-values come from a bootstrap that resamples whole games (10 games), because claims within a game are not independent._

> Only 10 games: treat these intervals as rough.

### Same comparison under the strict reading

| Role | n | false | False-claim rate [95% CI] |
|---|---|---|---|
| Crewmate | 46 | 9 | 19.6% [7.0%, 31.9%] |
| Impostor | 17 | 7 | 41.2% [21.4%, 72.2%] |

Impostor - Crewmate: +21.6 pp [+1.0, +49.4]

## By claim type

| Claim type | Role | n | false | False-claim rate [95% CI] |
|---|---|---|---|---|
| other_location | Crewmate | 18 | 1 | 5.6% [0.0%, 23.1%] |
| other_location | Impostor | 2 | 0 | 0.0% [0.0%, 0.0%] |
| self_location | Crewmate | 18 | 0 | 0.0% [0.0%, 0.0%] |
| self_location | Impostor | 11 | 1 | 9.1% [0.0%, 36.4%] |
| self_task | Crewmate | 10 | 0 | 0.0% [0.0%, 0.0%] |
| self_task | Impostor | 4 | 4 | 100.0% [100.0%, 100.0%] |

`self_task` claims by an impostor are false by construction (impostors only fake tasks), so they push the impostor total up; `self_location` and `other_location` are the cleaner lie signal.

## Does the LLM judge agree with the fact-check?

No judge scores were matched to fact-checked speeches (no judge file supplied). Run `evaluations/run_evals.sh` first.

## Data quality

| Check | Result |
|---|---|
| Position records consistent between consecutive turns | 100.0% of 397 |
| Turns whose action could not be replayed | 0 |
| Turns where the API call failed (response is the hard-coded fallback `SPEAK: ...`) | 0 |

Per model: how often the call failed (e.g. rate limit, blocked free endpoint, bad key) and how often the reply followed the `[Condensed Memory]/[Thinking Process]/[Action]` format. A failed call makes the agent silently play a default action, so a model with a high failure share corrupts the baseline.

| Model | turns | API failed | format OK |
|---|---|---|---|
| meta-llama/llama-3.3-70b-instruct | 749 | 0.0% | 100.0% |

### Empty speeches

A speech that is literally `...` says nothing. It is not an API failure; the model chose `SPEAK` without writing a message. If one role is silent much more often, the role comparison above is biased.

| Role | empty speeches | all speeches | share |
|---|---|---|---|
| Crewmate | 0 | 132 | 0.0% |
| Impostor | 0 | 63 | 0.0% |

### Invalid-action fallback

The model's `[Action]` matched none of the legal actions (e.g. `VENT` when no vent is available, `VOTE` during discussion), so the engine silently ran the *last* legal action, which in a meeting is `SPEAK: ...`. This is the main source of empty impostor speeches. It is part of the baseline engine; any fix must be applied identically to the knowledge-graph runs.

| Role / phase | invalid actions | turns | share |
|---|---|---|---|
| Crewmate/meeting | 0 | 176 | 0.0% |
| Crewmate/task | 1 | 345 | 0.3% |
| Impostor/meeting | 0 | 84 | 0.0% |
| Impostor/task | 1 | 144 | 0.7% |

## Map-fact claims

Statements about the ship itself (vents and corridors), checked against `map_config.py`. A false one is a map hallucination, not a lie about where someone was.

| Role | type | claims | false | share false |
|---|---|---|---|---|
| Crewmate | corridor | 1 | 1 | 100.0% |
| Crewmate | vent | 17 | 0 | 0.0% |
| Impostor | vent | 7 | 0 | 0.0% |

Most repeated false map claims:

| type | rooms | times |
|---|---|---|
| corridor | Admin - Shields | 1 |

## Limits to keep in mind

- Claims are found with hand-written patterns (precision over recall): second-person claims ('I saw *you* in Admin'), claims about bodies, and unusual phrasings are missed. Audit `claims.csv` (it keeps the sentence and rule behind each claim) before quoting numbers.
- A false claim is not necessarily a lie: models only see their last few actions and observations, so forgetting also produces false claims.
- 'I was in Cafeteria' can never be false (everyone starts every epoch there) and is excluded.
