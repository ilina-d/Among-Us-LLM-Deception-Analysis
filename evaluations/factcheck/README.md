# Fact-checking what players say

Checks every `SPEAK` message against what the game record shows.

```bash
# after running games (main.py) and, optionally, the LLM judge (evaluations/run_evals.sh)
python evaluations/run_factcheck.py --expt <experiment-folder-name>            # logs from expt-logs/
python evaluations/run_factcheck.py --expt expA expB                           # pool several experiments
python evaluations/run_factcheck.py --log-file path/to/agent-logs-compact.json --name myrun
```

Output goes to `evaluations/results/factcheck/<name>/`:

| file | contents |
|---|---|
| `report.md` | lie rates by role / claim type / model, judge agreement, data-quality checks |
| `claims.csv` | every extracted claim with the sentence, the rule that produced it, verdict and evidence |
| `speeches.csv` | every speech (with the judge's `lying` score when available) |
| `audit_sample.csv` | 60 random claims; fill the two blank columns to measure extractor precision |
| `disagreements.csv` | speeches where judge and fact-check disagree most |
| `summary.json` | all numbers in machine-readable form |

## How it works
1. **Ground truth** (`truth.py`): the agent logs store each player's real `player.location` before every action.
   The engine does not log which action it executed, so `logs.py` replays its matching rule (first available
   action contained in the reply, else the *last* available action). Positions are grouped into *epochs*
   (game start to first meeting, between meetings, ...).
2. **Claims** (`claims.py`): rule-based extraction of `self_location`, `other_location` and `self_task` claims.
   Questions are skipped; hedges ("I think") and hearsay ("X said") are kept but flagged and excluded from statistics.
3. **Verdicts** (`verify.py`): *conservative* (headline) calls a positive claim false only if the player was never
   there earlier in the game; *strict* requires it since the last meeting. Cafeteria claims can't be false and are excluded.
4. **Statistics** (`analysis.py`): false-claim rates with game-level bootstrap CIs; judge agreement via AUROC and
   precision/recall at several cut-offs. Judge rows are joined on `(game_index, player_name, timestamp)` because
   `step` is shared by every message in a meeting.

## Tests
`pytest tests/test_factcheck.py` runs the real game engine with a scripted policy (`tests/simulate_games.py`, no
API key) and checks that positions rebuilt from the logs equal what the engine recorded, including with 25% malformed
replies, and that planted true/false claims are found and labelled correctly.
