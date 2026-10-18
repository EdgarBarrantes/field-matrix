#!/usr/bin/env python3
"""Print the latest `cargo bench` results as a Markdown section for benches/RESULTS.md.

    benches/run.sh
    python3 benches/summarize.py   # prints a section for benches/RESULTS.md

`python3 benches/summarize.py <baseline>` prints a saved criterion baseline instead of the latest
run (the heading then names the current commit: edit it to the baseline's).

Reads criterion's estimates (median time per operation) from target/criterion and adds what the
numbers depend on: commit, Rust and ark-ff versions, and the CPU.
"""
import datetime
import json
import sys
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRITERION = ROOT / "target" / "criterion"
ORDER = [
    "multiply", "determinant", "naive_determinant_leibniz", "inverse", "solve", "naive_solve_cramer",
    "adjoint", "adjugate", "naive_adjugate", "pow_64", "naive_pow_64", "evaluate_matrix_polynomial",
]


def run(*args):
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cpu():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ark_ff_version():
    lock = (ROOT / "Cargo.lock").read_text()
    match = re.search(r'name = "ark-ff"\nversion = "([^"]+)"', lock)
    return match.group(1) if match else "unknown"


def human(ns):
    for unit, size in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= size:
            return f"{ns / size:.3g} {unit}"
    return f"{ns:.3g} ns"


def main():
    baseline = sys.argv[1] if len(sys.argv) > 1 else "new"
    results = {}
    for estimates in CRITERION.glob(f"*/*/{baseline}/estimates.json"):
        group, size = estimates.parts[-4], estimates.parts[-3]
        if size.isdigit():
            median = json.loads(estimates.read_text())["median"]["point_estimate"]
            results.setdefault(group, {})[int(size)] = median
    if not results:
        raise SystemExit("No results in target/criterion: run `cargo bench` first.")

    sizes = sorted({n for by_size in results.values() for n in by_size})
    commit = run("git", "log", "-1", "--format=%h %s")
    dirty = " (with uncommitted changes)" if run("git", "status", "--porcelain") else ""
    print(f"## {datetime.date.today()}: {commit}{dirty}\n")
    conditions = CRITERION / "conditions.txt"
    how = conditions.read_text().strip() if conditions.exists() else "run without benches/run.sh"
    print(f"{run('rustc', '--version')}, ark-ff {ark_ff_version()}, {cpu()}; {how}. Median time per")
    print("operation.\n")
    print("| Operation | " + " | ".join(f"n = {n}" for n in sizes) + " |")
    print("|---|" + "---|" * len(sizes))
    for group in sorted(results, key=lambda g: (ORDER.index(g) if g in ORDER else len(ORDER), g)):
        cells = [human(results[group][n]) if n in results[group] else "" for n in sizes]
        print(f"| {group} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    main()
