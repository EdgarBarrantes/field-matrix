# Naive against fast

How long each operation takes when computed from its definition (the `naive` module) and with the
method on `Matrix`, on the same n x n matrices over a 255-bit field (BLS12-381's scalar field).
Median time per operation.

| Operation | Naive version | Fast version | n | Naive | Fast | Fast is |
|---|---|---|---|---|---|---|
| Determinant | Leibniz formula (all permutations) | elimination | 4 | 2.05 µs | 7.08 µs | 3.5x slower |
| Determinant | Leibniz formula (all permutations) | elimination | 6 | 92.1 µs | 11.2 µs | 8.2x faster |
| Determinant | Leibniz formula (all permutations) | elimination | 8 | 7.2 ms | 15.8 µs | 455x faster |
| Determinant | cofactor expansion | elimination | 4 | 1.7 µs | 6.88 µs | 4.1x slower |
| Determinant | cofactor expansion | elimination | 6 | 54.8 µs | 11.1 µs | 5.0x faster |
| Determinant | cofactor expansion | elimination | 8 | 3.12 ms | 16.1 µs | 194x faster |
| Adjugate | cofactors | det(A) * A^-1 (kernels if singular) | 4 | 120 µs | 15.7 µs | 7.7x faster |
| Adjugate | cofactors | det(A) * A^-1 (kernels if singular) | 8 | 1.63 ms | 45.2 µs | 36x faster |
| Adjugate | cofactors | det(A) * A^-1 (kernels if singular) | 16 | 19.6 ms | 242 µs | 81x faster |
| Inverse | adjugate / det(A) | Gauss-Jordan elimination | 4 | 138 µs | 8.36 µs | 17x faster |
| Inverse | adjugate / det(A) | Gauss-Jordan elimination | 8 | 1.67 ms | 27.9 µs | 60x faster |
| Inverse | adjugate / det(A) | Gauss-Jordan elimination | 16 | 19.7 ms | 177 µs | 111x faster |
| Solve A x = b | Cramer's rule | Gauss-Jordan elimination | 4 | 37.6 µs | 7.67 µs | 4.9x faster |
| Solve A x = b | Cramer's rule | Gauss-Jordan elimination | 16 | 1.38 ms | 100 µs | 14x faster |
| Solve A x = b | Cramer's rule | Gauss-Jordan elimination | 64 | 133 ms | 5.07 ms | 26x faster |
| A^64 | 64 multiplications | square-and-multiply | 4 | 66.4 µs | 7.27 µs | 9.1x faster |
| A^64 | 64 multiplications | square-and-multiply | 16 | 5.6 ms | 567 µs | 9.9x faster |

At small sizes the naive version can win: a field inversion costs as much as dozens of
multiplications, and the definitions need few or none. `cargo run --example operation_counts`
shows the operation counts behind these numbers.

Measured 2026-10-18, field-matrix-utils 0.2.0, rustc 1.99.1, Intel(R) Core(TM) Ultra 7 155H; pinned to CPUs 1-2, governor performance, 96% idle before, other cores 97% idle during (ok).
To measure again:
`benches/run.sh`, then `python3 benches/summarize.py > benches/RESULTS.md`.
