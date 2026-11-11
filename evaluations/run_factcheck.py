#!/usr/bin/env python3
"""Fact-check what players said against what the game record shows.

Examples (run from anywhere):
  python evaluations/run_factcheck.py --expt 2025-02-01_phi_phi_100_games_v3
  python evaluations/run_factcheck.py --expt expA expB --judge-dir evaluations/results
  python evaluations/run_factcheck.py --log-file /path/to/agent-logs-compact.json --name myrun --no-judge

Writes to evaluations/results/factcheck/<name>/: report.md, claims.csv, speeches.csv,
summary.json, audit_sample.csv (random claims to hand-check the extractor) and, when judge
scores exist, disagreements.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from factcheck import analysis as A  # noqa: E402
from factcheck.logs import load_experiment, load_log_file  # noqa: E402
from factcheck.report import render_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expt", nargs="*", default=[], help="experiment folder name(s) under --logs-root")
    ap.add_argument("--logs-root", default=str(ROOT / "expt-logs"))
    ap.add_argument("--log-file", help="path to a single agent-logs-compact.json (instead of --expt)")
    ap.add_argument("--name", help="name for the output folder (default: the experiment name(s))")
    ap.add_argument("--judge-dir", default=str(HERE / "results"),
                    help="where eval.py wrote <expt>_all_skill_scores.json (default: evaluations/results)")
    ap.add_argument("--judge-file", help="explicit judge results file (single experiment only)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=list(A.ALL_KINDS), choices=list(A.ALL_KINDS),
                    help="claim types to include in the statistics")
    ap.add_argument("--threshold", type=int, default=6, help="judge score treated as 'says lying' in the headline")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    games = {}
    experiments = []
    if args.log_file:
        label = args.name or Path(args.log_file).parent.name
        games.update(load_log_file(args.log_file, label))
        experiments.append(label)
    for e in args.expt:
        games.update(load_experiment(args.logs_root, e))
        experiments.append(e)
    if not games:
        ap.error("give --expt or --log-file")
    name = args.name or "+".join(experiments)
    out = Path(args.out) if args.out else HERE / "results" / "factcheck" / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(games)} games; fact-checking speeches...")
    claims, speeches, quality = A.build_tables(games)
    speeches = A.attach_counts(speeches, claims, args.kinds)

    merged, judge_info, agreement = None, None, None
    if not args.no_judge and len(speeches):
        parts, infos = [], {}
        for e in experiments:
            jf = Path(args.judge_file) if args.judge_file else Path(args.judge_dir) / f"{e}_all_skill_scores.json"
            if jf.exists():
                m, info = A.merge_judge(speeches, A.load_judge(jf), e)
                parts.append(m)
                infos[e] = info
            else:
                infos[e] = f"no judge file at {jf}"
        if parts:
            merged = pd.concat(parts)
            judge_info = infos if len(infos) > 1 else next(iter(infos.values()))
            agreement = A.judge_agreement(merged, args.threshold)
        else:
            judge_info = infos

    map_df = quality.get("_map_claims")
    report, summary = render_report(name, claims, speeches, quality, args.kinds, judge_info, agreement, B=args.bootstrap)
    (out / "report.md").write_text(report, encoding="utf-8")
    claims.to_csv(out / "claims.csv", index=False)
    if map_df is not None and len(map_df):
        map_df.to_csv(out / "map_claims.csv", index=False)
    (merged if merged is not None else speeches).to_csv(out / "speeches.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    d = A.counted(claims, args.kinds)
    if len(d):
        cols = ["experiment", "game", "player", "identity", "kind", "subject", "room", "task", "negated",
                "verdict", "sentence", "rule"]
        sample = d.sample(min(60, len(d)), random_state=0)[cols].assign(extraction_correct="", verdict_correct="")
        sample.to_csv(out / "audit_sample.csv", index=False)
    if agreement and agreement.get("checked") is not None:
        A.disagreements(agreement["checked"], claims, args.kinds).to_csv(out / "disagreements.csv", index=False)
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
