# Changelog

## 0.2.0

A new API (borrowed, with errors instead of panics), ark-ff 0.6, more linear algebra, and the
parts for learning: each main operation also computed from its definition, operation counting,
benchmarks and worked examples. Upgrading from 0.1: see the table below.

Breaking:
- Rust 1.83 or newer (`rust-version` in `Cargo.toml`, checked in CI): what ark-ff 0.6 really needs,
  though it declares 1.75.
- ark-ff 0.6 (was 0.4): use field types from ark-ff 0.6. In ark-ff 0.5 and later, `ZERO` comes
  from the `AdditiveGroup` trait, so code using `F::ZERO` on a concrete field type needs
  `use ark_ff::AdditiveGroup`.
- Methods take `&self` instead of `self`: no more `a.clone().determinant()`. The operators work on
  owned matrices and on references (`&a + &b`, `&a * &b`, `-&a`).
- Methods that can fail return `Result<_, MatrixError>` instead of panicking:

  | 0.1 | 0.2 |
  |---|---|
  | `Matrix::new(rows)` (panics on rows of different lengths) | `Matrix::from_rows(rows)?` |
  | | `Matrix::new(rows, columns, entries)?` (entries row by row) |
  | `determinant()` | `determinant()?` (`NotSquare`) |
  | `inverse()` returning `Option` | `inverse()?` (`NotSquare`, `Singular`) |
  | `ax_b_solve_for_x(b)` | `solve(&b)?` (`NotSquare`, `DimensionMismatch`, `Singular`) |
  | `adjoint()` | `adjugate()?` (`NotSquare`) |
  | `lu_decomposition()` | `lu_decomposition()?` (`NotSquare`, `ZeroPivot`) |
  | `mul_vec(v)` | `mul_vec(&v)?` (`DimensionMismatch`) |
  | `sum_of_matrix()` | `sum()` |
  | `get_element(i, j)` | `get(i, j)` (`Option`) or `a[(i, j)]` |
  | `set_element(i, j, x)` | `set(i, j, x)?` (`OutOfBounds`) or `a[(i, j)] = x` |
  | `evaluate_*_polynomial` returning `Option` | `Result` (`Empty`, `DimensionMismatch`) |

  The operators `+`, `-`, `*` still panic on sizes that don't fit.
- Entries are stored in one `Vec`, row by row ([#1](https://github.com/EdgarBarrantes/field-matrix/issues/1)),
  so a matrix always has a shape: `rows()` and `columns()`, also for empty ones (a 2 x 0 matrix is
  not a 0 x 2 one).
- The empty 0 x 0 matrix is square, with determinant 1, and is its own inverse and identity
  (0.1 said it wasn't square).
- `lu_decomposition` returns `ZeroPivot` only when it would divide by zero: singular matrices
  whose last pivot is zero, such as `[[1, 2], [2, 4]]`, now have an LU decomposition (0.1 rejected
  every singular matrix).
- `Display` prints the entries as numbers, aligned by column (0.1 printed `BigInt([3])`).

New:
- `zeros`, `identity`, `from_fn`, `rows`, `columns`, `as_slice`, `to_rows`, indexing with
  `a[(i, j)]`, negation, and `MatrixError` (which implements `std::error::Error`).
- `naive`: the operations computed from their definitions, to compare with the methods:
  `determinant_leibniz`, `determinant_laplace`, `adjugate` (cofactors), `inverse` (adjugate /
  determinant), `solve_cramer`, `pow`.
- `counting`: `Counting<F>` counts the field operations done with it (`counting::count`), and
  `cargo run --example operation_counts` prints naive against fast by matrix size.
- `pow`, by square-and-multiply.
- `reduced_row_echelon_form`, `rank`, `kernel` (a basis of the null space) and `solve_all`: every
  solution of `A x = b` for a matrix of any shape (`None` when there is none), as one solution
  plus the kernel.
- `trace` and `characteristic_polynomial` (Berkowitz's algorithm, which never divides, so it works
  in small fields where Faddeev-LeVerrier would divide by zero).
- `Matrix::vandermonde`, `interpolate` (the polynomial through given points) and
  `evaluate_polynomial`, in `polynomial` and at the crate root.
- Worked examples (`cargo run --example <name>`): `shamir`, `reed_solomon`, `r1cs`, `hill_cipher`,
  and `operation_counts`.

Changed:
- `adjugate` no longer computes n^2 cofactors (about n^5 work): det(A) * A^-1 when invertible, a
  rank-one matrix from the kernels when the rank is n - 1, zero below; 80 times faster at n = 16.
  The cofactor version is `naive::adjugate`.
- `determinant` inverts each pivot once instead of dividing once per row: 33 to 78% faster.
- The matrix code works with entries implementing `scalar::Scalar`, which every `ark_ff::Field`
  does (and `Counting`); a `Matrix<F>` with `F: ark_ff::Field` works as before.

Other:
- Benchmarks (`cargo bench`; `benches/run.sh` for comparable runs: an idle machine, one
  performance core) with recorded results in `benches/RESULTS.md`, so changes in speed are
  visible over time.
- Tests: `tests/naive.rs` (naive against fast, pinned operation counts) and `tests/algebra.rs` (row
  reduction, kernel, solutions, Cayley-Hamilton, interpolation), over GF(7) and GF(127).
- CI also builds the library on Rust 1.83 and runs every example.

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
