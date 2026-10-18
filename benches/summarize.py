#!/usr/bin/env python3
"""Writes benches/RESULTS.md: each operation's naive version against the fast one.

    benches/run.sh
    python3 benches/summarize.py > benches/RESULTS.md

Reads criterion's median time per run from target/criterion.
"""
import datetime
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRITERION = ROOT / "target" / "criterion"

# Benchmark group, operation, and what the naive and the fast versions do.
OPERATIONS = [
    ("determinant_leibniz", "Determinant", "Leibniz formula (all permutations)", "elimination"),
    ("determinant_laplace", "Determinant", "cofactor expansion", "elimination"),
    ("adjugate", "Adjugate", "cofactors", "det(A) * A^-1 (kernels if singular)"),
    ("inverse", "Inverse", "adjugate / det(A)", "Gauss-Jordan elimination"),
    ("solve", "Solve A x = b", "Cramer's rule", "Gauss-Jordan elimination"),
    ("pow", "A^64", "64 multiplications", "square-and-multiply"),
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


def human(ns):
    for unit, size in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= size:
            return f"{ns / size:.3g} {unit}"
    return f"{ns:.3g} ns"


def median(group, version, n):
    estimates = CRITERION / group / version / str(n) / "new" / "estimates.json"
    if not estimates.exists():
        return None
    return json.loads(estimates.read_text())["median"]["point_estimate"]


def ratio(naive, fast):
    if naive >= fast:
        return f"{naive / fast:,.0f}x faster" if naive / fast >= 10 else f"{naive / fast:.1f}x faster"
    return f"{fast / naive:.1f}x slower"


def main():
    rows = []
    for group, name, naive_how, fast_how in OPERATIONS:
        sizes = sorted(int(p.name) for p in (CRITERION / group / "fast").glob("*") if p.name.isdigit())
        for n in sizes:
            naive, fast = median(group, "naive", n), median(group, "fast", n)
            if naive is not None and fast is not None:
                rows.append(
                    f"| {name} | {naive_how} | {fast_how} | {n} | {human(naive)} | {human(fast)} "
                    f"| {ratio(naive, fast)} |"
                )
    if not rows:
        raise SystemExit("No results in target/criterion: run benches/run.sh first.")

    conditions = CRITERION / "conditions.txt"
    how = conditions.read_text().strip() if conditions.exists() else "run without benches/run.sh"
    version = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)
    rustc = run("rustc", "--version").split(" (")[0]
    print(f"""# Naive against fast

How long each operation takes when computed from its definition (the `naive` module) and with the
method on `Matrix`, on the same n x n matrices over a 255-bit field (BLS12-381's scalar field).
Median time per operation.

| Operation | Naive version | Fast version | n | Naive | Fast | Fast is |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

At small sizes the naive version can win: a field inversion costs as much as dozens of
multiplications, and the definitions need few or none. `cargo run --example operation_counts`
shows the operation counts behind these numbers.

Measured {datetime.date.today()}, field-matrix-utils {version}, {rustc}, {cpu()}; {how}.\nTo measure again:
`benches/run.sh`, then `python3 benches/summarize.py > benches/RESULTS.md`.""")


if __name__ == "__main__":
    main()
