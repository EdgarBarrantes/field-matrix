# Changelog

## 0.1.2

Fixes wrong results and panics. If you used 0.1.1 or earlier, check results from the functions below.

Wrong results:
- `ax_b_solve_for_x` returned wrong solutions (back substitution used the wrong variable). It now uses
  Gaussian elimination with row exchanges.
- `mul_vec` computed v * A (the vector as a row) although documented as A * v. It now computes A * v, so
  the vector needs one element per column.
- `adjoint` returned the transpose. It now returns the adjugate (transposed cofactor matrix), so
  `a * a.adjoint() == det(a) * I`.

Panics on valid input:
- `inverse` and `ax_b_solve_for_x` panicked on invertible matrices with a zero pivot, such as
  [[0, 1], [1, 0]]. `inverse` now returns `None` when the matrix is not invertible instead of panicking.
- `scalar_mul` panicked on non-square matrices.
- `==` panicked on non-square matrices, and called matrices of different sizes equal when the top-left
  corner matched.
- `is_square` and other functions panicked on an empty matrix.

Other:
- `Matrix::new` rejects rows of different lengths.
- `lu_decomposition` (no row exchanges) explains when a matrix needs them instead of failing with a
  division by zero.
- Tests compare every operation with its definition on random matrices over two fields.

## 0.1.1

First release.
