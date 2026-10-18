# Benchmark results

How long each operation takes, recorded over time so a change shows whether it made things faster
or slower. Newest first. All runs on the same machine unless noted; timings from different
machines aren't comparable.

The field is the BLS12-381 scalar field (255 bits). Inputs are fixed pseudo-random n x n matrices
(`benches/operations.rs`). The adjugate is measured only up to n = 16: it is computed from cofactors
(about n^5 field multiplications).

To add a run (after committing):

```
benches/run.sh                 # waits until the machine is idle, pins to one performance core
python3 benches/summarize.py   # prints a section; paste it below this line
```

Runs before 2026-10-18 evening were done without `benches/run.sh` (unpinned, `powersave` governor):
compare them only with each other.

To compare a change with the previous run directly, criterion keeps the last results in
`target/criterion`: `cargo bench -- --save-baseline before`, change the code, then
`cargo bench -- --baseline before` reports the difference for each benchmark (pass the same
options to `benches/run.sh`).

## 2026-10-18: 45470c2 0.2 API: flat storage, &self, Result errors, new names

rustc 1.91.1 (ed61e7d7e 2025-11-07), ark-ff 0.6.0, Intel(R) Core(TM) Ultra 7 155H. Median time per operation.

| Operation | n = 4 | n = 8 | n = 16 | n = 64 |
|---|---|---|---|---|
| multiply | 990 ns |  | 63.7 µs | 5.64 ms |
| determinant | 6.51 µs |  | 51.1 µs | 2.15 ms |
| inverse | 7.88 µs |  | 173 µs | 9.95 ms |
| solve | 7.2 µs |  | 99.2 µs | 5.28 ms |
| adjugate | 117 µs | 1.64 ms | 19.8 ms |  |
| evaluate_matrix_polynomial | 2.69 µs |  | 35 µs | 617 µs |

The 0.2 API (flat storage, `&self`, `Result`), against the first run: multiply 2 to 14% faster,
determinant 33 to 78% faster (it now inverts each pivot once instead of dividing once per row),
inverse 2 to 9% and solve up to 14% faster (n = 64 unchanged), polynomial evaluation 20 to 25%
faster. The adjugate, renamed from `adjoint`, is 2.8 times faster at n = 16 but slower at n = 4
(117 against 85 µs), not explained yet; step 4 adds a fast adjugate, and the operation counts
should show where the time goes.
Measured after hades went idle (an earlier run during a Paperless import was off by up to 3x).

## 2026-10-18: 9a613db Benchmarks

rustc 1.91.1 (ed61e7d7e 2025-11-07), ark-ff 0.6.0, Intel(R) Core(TM) Ultra 7 155H. Median time per operation.

| Operation | n = 4 | n = 8 | n = 16 | n = 64 |
|---|---|---|---|---|
| multiply | 1.11 µs |  | 71.3 µs | 5.72 ms |
| determinant | 9.66 µs |  | 229 µs | 5.17 ms |
| inverse | 8.12 µs |  | 190 µs | 10.2 ms |
| solve | 7.41 µs |  | 115 µs | 5.2 ms |
| adjoint | 84.5 µs | 2.83 ms | 56.7 ms |  |
| evaluate_matrix_polynomial | 3.55 µs |  | 43.6 µs | 797 µs |

First recorded run: the 0.1 code as it is, on ark-ff 0.6. The baseline for the 0.2.0 changes. (0.1.4,
on ark-ff 0.4, can still be measured from its tag.)

