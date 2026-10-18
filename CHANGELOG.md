# Changelog

## Unreleased (0.2.0)

Breaking:
- ark-ff 0.6 (was 0.4): use field types from ark-ff 0.6. In ark-ff 0.5 and later, `ZERO` comes
  from the `AdditiveGroup` trait, so code using `F::ZERO` on a concrete field type needs
  `use ark_ff::AdditiveGroup`.

Other:
- Benchmarks (`cargo bench`, `benches/`) with recorded results in `benches/RESULTS.md`, so changes
  in speed are visible over time.

## 0.1.4

- `evaluate_matrix_polynomial` and `evaluate_vector_polynomial`: evaluate a polynomial whose
  coefficients are matrices or vectors at a point (`M_0 + alpha * M_1 + ... + alpha^k * M_k`), as
  used to batch checks with a random linear combination. From
  [#6](https://github.com/EdgarBarrantes/field-matrix/pull/6) by Edgar Barrantes and Isaac
  Villalobos.
- The matrix code lives in its own module: `Matrix` is also available as
  `field_matrix_utils::matrix::Matrix`.

## 0.1.3

Documentation only:
- Every example in the docs is now compiled and checked by `cargo test`; the crate example was
  missing a step and didn't compile.
- README and crate docs say who the crate is for, what it isn't for (not constant time, small dense
  matrices only) and the alternatives.
- License files (MIT and Apache-2.0, as already declared in `Cargo.toml`).

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
