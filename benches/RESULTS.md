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

## 2026-10-18: ee78a7f, same code, Rust 1.99.1 (was 1.91.1)

rustc 1.99.1 (58cf8f9c3 2026-10-13), ark-ff 0.6.0, Intel(R) Core(TM) Ultra 7 155H; pinned to CPUs 1-2, governor performance, 95% idle before, other cores 96% idle during (ok). Median time per
operation.

| Operation | n = 4 | n = 8 | n = 16 | n = 64 |
|---|---|---|---|---|
| multiply | 1.07 µs |  | 64.9 µs | 5.7 ms |
| determinant | 6.98 µs |  | 57.2 µs | 2.01 ms |
| naive_determinant_leibniz | 1.49 µs | 5.29 ms |  |  |
| inverse | 8.43 µs |  | 182 µs | 9.81 ms |
| solve | 7.64 µs |  | 108 µs | 5.14 ms |
| naive_solve_cramer | 41.8 µs |  | 1.41 ms |  |
| adjugate | 15.8 µs | 45.2 µs | 249 µs |  |
| naive_adjugate | 127 µs | 1.64 ms | 19.8 ms |  |
| pow_64 |  |  | 557 µs |  |
| naive_pow_64 |  |  | 5.68 ms |  |
| rank | 7.14 µs |  | 78.9 µs | 3.01 ms |
| characteristic_polynomial | 1.63 µs |  | 322 µs | 86.7 ms |
| evaluate_matrix_polynomial | 2.33 µs |  | 36.3 µs | 620 µs |

Only the compiler changed (1.91.1 to 1.99.1) against the ee78a7f run above it. Mostly slower:
elimination-based operations (determinant, inverse, solve, rank) about 10% at n = 4, 4 to 23% at
n = 16 (rank 23%, determinant 20%, the noisy one) and 2 to 5% at n = 64 (determinant 4% faster);
multiply 2 to 5%; polynomial evaluation and `pow` unchanged. Consistent in direction and the run
was clean, so a compiler effect, not noise; nothing to fix in the crate.

## 2026-10-18: 9541232 Rank, kernel, all solutions, trace, characteristic polynomial, interpolation

rustc 1.91.1 (ed61e7d7e 2025-11-07), ark-ff 0.6.0, Intel(R) Core(TM) Ultra 7 155H; pinned to CPUs 1-2, governor performance, 97% idle before, other cores 97% idle during (ok). Median time per
operation.

| Operation | n = 4 | n = 8 | n = 16 | n = 64 |
|---|---|---|---|---|
| multiply | 1.05 µs |  | 63.6 µs | 5.43 ms |
| determinant | 6.25 µs |  | 47.4 µs | 2.09 ms |
| naive_determinant_leibniz | 1.41 µs | 5.01 ms |  |  |
| inverse | 7.72 µs |  | 173 µs | 9.63 ms |
| solve | 6.9 µs |  | 102 µs | 5 ms |
| naive_solve_cramer | 34 µs |  | 1.36 ms |  |
| adjugate | 14.5 µs | 42.3 µs | 241 µs |  |
| naive_adjugate | 113 µs | 1.56 ms | 19.2 ms |  |
| pow_64 |  |  | 548 µs |  |
| naive_pow_64 |  |  | 5.51 ms |  |
| rank | 6.43 µs |  | 64 µs | 2.95 ms |
| characteristic_polynomial | 1.71 µs |  | 305 µs | 85.1 ms |
| evaluate_matrix_polynomial | 2.34 µs |  | 36.4 µs | 623 µs |

New: `rank` costs about what `determinant` does (both one elimination). `characteristic_polynomial`
(Berkowitz, about n^4) beats the determinant at n = 4, as it never inverts, and is 40 times slower
at n = 64. The operations step 5 didn't change match the 6aa27d6 run within about 1%, except the
determinant at n = 16, 9% faster with the same code: run-to-run variation, even pinned, so the
+3.6% noted for it at 6aa27d6 is probably the same.

## 2026-10-18: 6aa27d6 Naive versions, operation counting, fast adjugate, pow

rustc 1.91.1 (ed61e7d7e 2025-11-07), ark-ff 0.6.0, Intel(R) Core(TM) Ultra 7 155H; pinned to CPUs 1-2, governor performance, 97% idle before, other cores 97% idle during (ok). Median time per
operation.

| Operation | n = 4 | n = 8 | n = 16 | n = 64 |
|---|---|---|---|---|
| multiply | 1.03 µs |  | 63.8 µs | 5.44 ms |
| determinant | 6.28 µs |  | 52 µs | 2.07 ms |
| naive_determinant_leibniz | 1.42 µs | 5.05 ms |  |  |
| inverse | 7.69 µs |  | 171 µs | 9.57 ms |
| solve | 6.92 µs |  | 98.7 µs | 5.01 ms |
| naive_solve_cramer | 33.6 µs |  | 1.37 ms |  |
| adjugate | 14.4 µs | 42.3 µs | 241 µs |  |
| naive_adjugate | 112 µs | 1.55 ms | 19.2 ms |  |
| pow_64 |  |  | 550 µs |  |
| naive_pow_64 |  |  | 5.49 ms |  |
| evaluate_matrix_polynomial | 2.34 µs |  | 36.4 µs | 617 µs |

Against the 45470c2 re-run below, under the same conditions: `adjugate` (now det * A^-1, or rank
one when singular) is 87% faster at n = 4, 97% at n = 8 and 98.7% at n = 16 (80 times). Unchanged
operations are within 1.5%, except the determinant at n = 16 (+3.6%) and polynomial evaluation
(+6%), whose code only moved from `ark_ff::Field` to the `Scalar` bound: not explained yet.

The naive versions: `naive_adjugate` is the old `adjugate` and times the same. The Leibniz
determinant beats elimination at n = 4 (no inversions; four inversions in a 255-bit field cost more
than 96 multiplications) and loses by 2,400 times at n = 8. Cramer's rule is 14 times slower than
`solve` at n = 16; `A^64` by square-and-multiply 10 times faster than 64 multiplications.

## 2026-10-18: 45470c2 0.2 API (re-run with benches/run.sh)

rustc 1.91.1 (ed61e7d7e 2025-11-07), ark-ff 0.6.0, Intel(R) Core(TM) Ultra 7 155H; pinned to CPUs 1-2, governor performance, 98% idle before, other cores 96% idle during (ok). Median time per
operation.

| Operation | n = 4 | n = 8 | n = 16 | n = 64 |
|---|---|---|---|---|
| multiply | 1.03 µs |  | 63.5 µs | 5.45 ms |
| determinant | 6.26 µs |  | 50.3 µs | 2.06 ms |
| inverse | 7.66 µs |  | 168 µs | 9.46 ms |
| solve | 6.93 µs |  | 97.9 µs | 4.94 ms |
| adjugate | 112 µs | 1.56 ms | 19.3 ms |  |
| evaluate_matrix_polynomial | 2.21 µs |  | 34.2 µs | 580 µs |

The 0.2 API code again, pinned to one performance core with the `performance` governor: the
baseline for the step after it. Against the earlier unpinned run of the same code, timings are
mostly 1 to 6% lower (polynomial evaluation at n = 4: 18% lower; multiply at n = 4: 4% higher), and
much steadier (confidence intervals within about 0.5%).

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

