# field-matrix-utils

Simple matrix library for Rust.
Used for use with Finite fields.

Not safe for production use.
It was only done for educational purposes.

## Example

```rust
#[derive(ark_ff::MontConfig)]
#[modulus = "127"]
#[generator = "6"]
pub struct F127Config;
type F = Fp64<MontBackend<F127Config, 1>>;
let a: Matrix<F> = Matrix::new(vec![
    vec![F::from(1), F::from(2)],
    vec![F::from(3), F::from(4)],
]);
let b = a.transpose();
let c = a + b;
let d = a * b;
let det = a.determinant();
```

## Some features include:

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
