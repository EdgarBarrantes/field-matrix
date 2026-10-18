# Benchmark results

How long each operation takes, recorded over time so a change shows whether it made things faster
or slower. Newest first. All runs on the same machine unless noted; timings from different
machines aren't comparable.

The field is the BLS12-381 scalar field (255 bits). Inputs are fixed pseudo-random n x n matrices
(`benches/operations.rs`). The adjugate is measured only up to n = 16: it is computed from cofactors
(about n^5 field multiplications).

To add a run (on an idle machine, after committing):

```
cargo bench
python3 benches/summarize.py   # prints a section; paste it below this line
```

To compare a change with the previous run directly, criterion keeps the last results in
`target/criterion`: `cargo bench -- --save-baseline before`, change the code, then
`cargo bench -- --baseline before` reports the difference for each benchmark.

