# field-matrix-utils

[![crates.io](https://img.shields.io/crates/v/field-matrix-utils.svg)](https://crates.io/crates/field-matrix-utils)
[![docs.rs](https://img.shields.io/docsrs/field-matrix-utils)](https://docs.rs/field-matrix-utils)
[![Tests](https://github.com/EdgarBarrantes/field-matrix/actions/workflows/cargo.yml/badge.svg)](https://github.com/EdgarBarrantes/field-matrix/actions/workflows/cargo.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/field-matrix-utils.svg)](#license)

Dense matrices over finite fields, for Rust: arithmetic, determinant, inverse, adjugate, LU
decomposition and linear systems (Ax = b), with any [arkworks](https://arkworks.rs) field
(`ark_ff::Field`, ark-ff 0.6) as the element type.

Exact arithmetic modulo a prime has no rounding, so a matrix inverse or a solved system is
exactly right or the matrix is singular. That is what you need for exercises in cryptography,
coding theory and zero-knowledge proofs, where general-purpose matrix crates such as nalgebra
don't fit: their inverse and LU need real or complex numbers.

## Who it's for

- Learning and prototyping: courses, exercises and toy versions of schemes that need small
  dense linear algebra over a prime field (secret sharing, erasure codes, R1CS by hand).
- Code that already uses arkworks 0.6 field types and needs a determinant, inverse or solve
  without writing Gaussian elimination again.

## Not for

- Production or secret data: it is not constant time (it branches on the values) and it has
  not been audited.
- Large or sparse matrices: dense storage, operations of about n^3 (the characteristic
  polynomial n^4) written for clarity, and no parallelism; see [benches/RESULTS.md](benches/RESULTS.md) for timings.
- Matrices over GF(2): [bitgauss](https://crates.io/crates/bitgauss) is much faster there. In
  Python, [galois](https://github.com/mhostetter/galois) and SageMath do all of this and more.

## Example

```rust
use ark_ff::{Fp64, MontBackend};
use field_matrix_utils::{Matrix, MatrixError};

// A prime field: the integers modulo 127. Any arkworks field works.
#[derive(ark_ff::MontConfig)]
#[modulus = "127"]
#[generator = "6"]
pub struct F127Config;
type F = Fp64<MontBackend<F127Config, 1>>;

let a: Matrix<F> = Matrix::from_rows(vec![
    vec![F::from(1), F::from(2)],
    vec![F::from(3), F::from(4)],
])?;

let product = &a * &a.transpose();
assert_eq!(product[(0, 0)], F::from(5));

// 1 * 4 - 2 * 3 = -2, that is 125 modulo 127
assert_eq!(a.determinant()?, -F::from(2));

let inverse = a.inverse()?;
assert!((&a * &inverse).is_identity());

// Solve a * x = [1, 2]
let b = vec![F::from(1), F::from(2)];
let x = a.solve(&b)?;
assert_eq!(a.mul_vec(&x)?, b);

println!("{a}"); // [1, 2]
                 // [3, 4]
```

## Features

- Construction: `new` (rows, columns, entries), `from_rows`, `zeros`, `identity`, `from_fn`
- `+`, `-`, `*` and unary `-` on matrices and references, `==`, `Display`, `a[(i, j)]`
- `scalar_mul`, `mul_vec` (A * v), `sum`, `transpose`, `get`, `set`
- `determinant`, `inverse`, `adjugate`, `solve` (Gaussian elimination with row exchanges) and
  `lu_decomposition` (Doolittle, without row exchanges)
- `reduced_row_echelon_form`, `rank`, `kernel`, `solve_all` (every solution of `A x = b`, any
  shape), `trace`, `characteristic_polynomial` (Berkowitz, division-free)
- `pow` (square-and-multiply), `is_square`, `is_diagonal`, `is_identity`
- `evaluate_matrix_polynomial` and `evaluate_vector_polynomial`: `M_0 + alpha * M_1 + ... +
  alpha^k * M_k`, the random linear combination proof systems use to batch checks
- `Matrix::vandermonde`, `polynomial::interpolate` and `polynomial::evaluate_polynomial`

Methods that can fail return a `MatrixError`; the operators panic if the sizes don't fit.

## Learning from it

- `naive` computes the same operations straight from their definitions (the Leibniz formula,
  cofactor expansion, the adjugate from cofactors, Cramer's rule...), next to the fast methods
  on `Matrix`. Tests check that both always agree.
- `counting` counts the field operations an algorithm does, independent of the machine:
  wrap a matrix's entries and run the operation inside `counting::count`.
  `cargo run --example operation_counts` prints a table of both versions by matrix size.
- `cargo bench` measures the time; [benches/RESULTS.md](benches/RESULTS.md) keeps the results over time.
- Worked examples in [`examples/`](examples), each runnable with `cargo run --example <name>`:
  `shamir` (secret sharing by interpolation), `reed_solomon` (recovering lost symbols, detecting
  a changed one), `r1cs` (checking a zero-knowledge constraint system, and batching its checks),
  `hill_cipher` (a matrix cipher, and breaking it with an inverse).

## Upgrading from 0.1

0.2 uses ark-ff 0.6, takes `&self` instead of `self` (no more `.clone()` before every call),
returns `Result` instead of panicking, and renames a few methods: `ax_b_solve_for_x` is `solve`,
`adjoint` is `adjugate`, `sum_of_matrix` is `sum`, `get_element` / `set_element` are `get` /
`set`, and `Matrix::new` takes the size and the entries (`from_rows` takes rows). The full list
is in the [changelog](https://github.com/EdgarBarrantes/field-matrix/blob/master/CHANGELOG.md).

Versions 0.1.1 and earlier gave wrong results; upgrade from those in any case.

## Install

```toml
[dependencies]
field-matrix-utils = "0.2"
ark-ff = "0.6"
```

## Development

`cargo test` runs the unit tests, the doc examples and `tests/random.rs`, which checks every
operation against its definition on random matrices over two fields (the determinant by the
permutation formula, `A * inverse == I`, `A * adjugate == det * I`, `L * U == A`, `A * x == b`...).

`cargo bench` times the main operations; [benches/RESULTS.md](benches/RESULTS.md) keeps the
results over time and says how to add a run.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
