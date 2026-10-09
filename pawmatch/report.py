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


def m4_table(results):
    r = results.get("m4_latency.json")
    if not r:
        return "No C++ benchmark yet."
    rows = ", ".join(f"{int(size):,} rows: {g['real']:,} real and {g['synthetic']:,} synthetic"
                     for size, g in r["gallery_rows"].items())
    lines = [f"{r['hardware']}, {r['threads']} thread, top {r['k']}, one query at a time. Each figure is the median "
             f"across {r['rounds']} rounds of that round's percentile over {r['timed_queries']:,} queries. "
             f"Gallery {rows}; synthetic rows are random unit vectors, and brute-force latency does not "
             f"depend on vector values. Real rows are {r['vectors']['source']}.", "",
             "| Gallery | Version | p50 (us) | p99 (us) | p99.9 (us) | Heap allocations per query |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {s['gallery']:,} | {s['version']} | {s['p50_us']:.1f} | {s['p99_us']:.1f} | {s['p999_us']:.1f} | "
              f"{s['heap_allocations_per_query']:g} |" for s in r["summary"]]
    return "\n".join(lines)


def problem(results):
    e = results.get("external.json")
    if not e:
        return "Problem statement pending sourced figures."
    intake, rto, chip = e["shelter_intakes_2025"], e["returned_to_owner_2025"], e["microchip_return_to_owner_dogs"]
    return (f"US shelters and rescues took in an estimated {intake['as_published']} dogs and cats in 2025 and returned "
            f"{rto['as_published']} of them to their owners (Shelter Animals Count, 2025 Annual Data Report). In a "
            f"2009 study of 53 shelters, microchipped stray dogs went home {100 * chip['microchipped_stray_dogs']:.1f}% "
            f"of the time, against {100 * chip['comparison_stray_dogs']:.1f}% for the comparison group of stray dogs "
            f"(Lord et al., J Am Vet Med Assoc 2009). Sources and caveats are in `results/external.json`.")


SECTIONS = {"problem": problem, "m1": m1_table, "m4": m4_table}
EXTERNAL = "external.json"  # sourced outside figures: shown in the README, never offered as project results


def readme(path="README.md", root="results"):
    text, results = Path(path).read_text(), load_results(root)
    for name, table in SECTIONS.items():
        text = re.sub(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)",
                      lambda m: m[1] + table(results) + "\n" + m[2], text, flags=re.S)
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
    return [f"{name}: {k} = {v}" for name, r in load_results(root).items() if name != EXTERNAL for k, v in walk(r, "")]


if __name__ == "__main__":
    if sys.argv[1] == "readme":
        readme()
    else:
        print("\n".join(facts()) or "No results yet.")
