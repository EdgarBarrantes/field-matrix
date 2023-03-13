# field-matrix-utils

[![crates.io](https://img.shields.io/crates/v/field-matrix-utils.svg)](https://crates.io/crates/field-matrix-utils)
![Tests](https://github.com/EdgarBarrantes/field-matrix/actions/workflows/cargo.yml/badge.svg)

Simple matrix library for Rust.
Used for use with Finite fields.

Not safe for production use.
It was only done for educational purposes.

## Example

```rust
// Arkworks has a macro to generate the modulus and generator for a finite field.
// Type F is field element for use in our matrix.
// You should be able to use any. This is just an example.
use ark_ff::{Fp64, MontBackend};
#[derive(ark_ff::MontConfig)]
#[modulus = "127"]
#[generator = "6"]
pub struct F127Config;
type F = Fp64<MontBackend<F127Config, 1>>;

// The good stuff starts here.
let a: Matrix<F> = Matrix::new(vec![
    vec![F::from(1), F::from(2)],
    vec![F::from(3), F::from(4)],
]);
let b: Matrix<F> = a.transpose();
let c: Matrix<F> = a + b;
let d: Matrix<F> = a * b;
let det: F = a.determinant();
...
```

## Features:

- Addition
- Subtraction
- Multiplication
- Transpose
- Determinant
- Inverse
- Is square
- Adjoint
- LU decomposition
- Scalar multiplication
- Vector multiplication
- Sumation
- Get element at index
- Set element at index
- Is identity
- Equality
- Display
- Linear equations Ax = b for x solution

License: MIT OR Apache-2.0
