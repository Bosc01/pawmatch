"""README tables and resume-bullet facts, built only from results/*.json.

results/dev/ holds DogFaceNet development runs and is never read here. The glob
below is not recursive, and tests/test_report.py fails if that changes.

Usage: python -m pawmatch.report readme|facts
"""
import json
import re
import sys
from pathlib import Path


def load_results(root="results"):
    return {p.name: json.loads(p.read_text()) for p in sorted(Path(root).glob("*.json"))}


def _pct(m):
    lo, hi = m["ci95"]
    return f"{100 * m['value']:.2f}% ({100 * lo:.2f} to {100 * hi:.2f})"


def m1_table(results):
    rows = [r for r in results.values() if "closed_set" in r]
    if not rows:
        return "No PetFace results yet."
    lines = ["| Model | Rank-1 | Rank-5 | mAP | Verification AUC (paper method) |", "|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {_pct(r['closed_set']['rank1'])} | {_pct(r['closed_set']['rank5'])} | "
              f"{_pct(r['map_leave_one_out'])} | {_pct(r['verification']['auc_clamped_as_paper'])} |" for r in rows]
    return "\n".join(lines)


def readme(path="README.md", root="results"):
    text = Path(path).read_text()
    table = m1_table(load_results(root))
    text = re.sub(r"(<!-- results:start -->\n).*?(<!-- results:end -->)",
                  lambda m: m[1] + table + "\n" + m[2], text, flags=re.S)
    Path(path).write_text(text)


def facts(root="results"):
    """Every number available to resume bullets, with the file it comes from."""
    def walk(x, key):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from walk(v, f"{key}.{k}" if key else k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from walk(v, f"{key}[{i}]")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            yield key, x
    return [f"{name}: {k} = {v}" for name, r in load_results(root).items() for k, v in walk(r, "")]


if __name__ == "__main__":
    if sys.argv[1] == "readme":
        readme()
    else:
        print("\n".join(facts()) or "No results yet.")
