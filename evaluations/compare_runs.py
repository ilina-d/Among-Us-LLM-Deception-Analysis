"""Compare two fact-check runs (e.g. baseline vs knowledge-graph memory).

    python evaluations/compare_runs.py baseline_out/summary.json kg_out/summary.json [--labels baseline kg]

Prints a side-by-side table of every numeric metric found in both summary.json files
(lie rates, empty speeches, invalid-action fallback, map-fact errors, judge agreement). With few games the
differences are noise; the game counts are shown first so you can judge that.
"""
import argparse
import json


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        if len(d) == 2 and all(isinstance(x, (int, float)) for x in d) and d[1]:
            out[prefix[:-1] + " (share)"] = d[0] / d[1]
        elif all(isinstance(x, dict) for x in d):
            for i, x in enumerate(d):
                tag = ",".join(str(x[k]) for k in ("identity", "kind", "role", "threshold") if k in x) or str(i)
                out.update(flat(x, f"{prefix}[{tag}]."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = float(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--only", default="", help="substring filter on metric names")
    args = ap.parse_args()
    fa, fb = (flat(json.load(open(p))) for p in (args.a, args.b))
    keys = [k for k in fa if k in fb and args.only in k and "format_ok_by_model" not in k and "turns_by_model" not in k]
    la, lb = args.labels
    print(f"{'metric':<70}{la:>12}{lb:>12}{'diff':>12}")
    for k in sorted(keys, key=lambda k: (not k.startswith("quality.games"), k)):
        print(f"{k[:69]:<70}{fa[k]:>12.3f}{fb[k]:>12.3f}{fb[k]-fa[k]:>+12.3f}")
    only = sorted(set(fa) ^ set(fb))
    if only:
        print(f"\n({len(only)} metrics present in only one run, e.g. a role with no claims; omitted)")


if __name__ == "__main__":
    main()
