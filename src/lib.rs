//! Dense matrices over finite fields, for Rust: arithmetic, determinant, inverse, adjugate, LU
//! decomposition and linear systems (Ax = b), with any [arkworks](https://arkworks.rs) field
//! (`ark_ff::Field`, ark-ff 0.6) as the element type.
//!
//! Exact arithmetic modulo a prime has no rounding, so a matrix inverse or a solved system is
//! exactly right or the matrix is singular. That is what you need for exercises in cryptography,
//! coding theory and zero-knowledge proofs, where general-purpose matrix crates such as nalgebra
//! don't fit: their inverse and LU need real or complex numbers.
//!
//! ## Who it's for
//!
//! - Learning and prototyping: courses, exercises and toy versions of schemes that need small
//!   dense linear algebra over a prime field (secret sharing, erasure codes, R1CS by hand).
//! - Code that already uses arkworks 0.6 field types and needs a determinant, inverse or solve
//!   without writing Gaussian elimination again.
//!
//! ## Not for
//!
//! - Production or secret data: it is not constant time (it branches on the values) and it has
//!   not been audited.
//! - Large or sparse matrices: rows are `Vec<Vec<F>>`, operations are O(n^3) (the adjugate is
//!   O(n^5)), and errors panic instead of returning `Result`.
//! - Matrices over GF(2): [bitgauss](https://crates.io/crates/bitgauss) is much faster there. In
//!   Python, [galois](https://github.com/mhostetter/galois) and SageMath do all of this and more.
//!
//! ## Example
//!
//! ```rust
//! use ark_ff::{Fp64, MontBackend};
//! use field_matrix_utils::Matrix;
//!
//! // A prime field: the integers modulo 127. Any arkworks field works.
//! #[derive(ark_ff::MontConfig)]
//! #[modulus = "127"]
//! #[generator = "6"]
//! pub struct F127Config;
//! type F = Fp64<MontBackend<F127Config, 1>>;
//!
//! let a: Matrix<F> = Matrix::new(vec![
//!     vec![F::from(1), F::from(2)],
//!     vec![F::from(3), F::from(4)],
//! ]);
//!
//! // Methods take the matrix by value: clone it to keep using it.
//! let b = a.clone().transpose();
//! let product = a.clone() * b;
//! assert_eq!(product.get_element(0, 0), F::from(5));
//!
//! // 1 * 4 - 2 * 3 = -2, that is 125 modulo 127
//! assert_eq!(a.clone().determinant(), -F::from(2));
//!
//! let inverse = a.clone().inverse().expect("the determinant is not zero");
//! assert!((a.clone() * inverse).is_identity());
//!
//! // Solve a * x = [1, 2]
//! let x = a.clone().ax_b_solve_for_x(vec![F::from(1), F::from(2)]);
//! assert_eq!(a.mul_vec(x), vec![F::from(1), F::from(2)]);
//! ```
//!
//! ## Features
//!
//! - `+`, `-`, `*` between matrices, `==`, and `Display`
//! - `scalar_mul`, `mul_vec` (A * v), `sum_of_matrix`, `transpose`
//! - `determinant`, `inverse` (`None` when singular), `adjoint` (the adjugate)
//! - `ax_b_solve_for_x` (Gaussian elimination with row exchanges) and `lu_decomposition`
//!   (Doolittle, without row exchanges)
//! - `is_square`, `is_diagonal`, `is_identity`, `get_element`, `set_element`
//! - `evaluate_matrix_polynomial` and `evaluate_vector_polynomial`: `M_0 + alpha * M_1 + ... +
//!   alpha^k * M_k`, the random linear combination proof systems use to batch checks
//!
//! ## Upgrading from 0.1.1 or earlier
//!
//! Those versions gave wrong results: `ax_b_solve_for_x` returned wrong solutions, `mul_vec`
//! computed v * A instead of A * v, and `adjoint` returned the transpose. They also panicked on
//! valid input. Details in the
//! [changelog](https://github.com/EdgarBarrantes/field-matrix/blob/master/CHANGELOG.md).

pub mod matrix;
pub mod polynomial;

pub use matrix::Matrix;
pub use polynomial::{evaluate_matrix_polynomial, evaluate_vector_polynomial};
