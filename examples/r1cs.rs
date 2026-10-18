//! A rank-1 constraint system (R1CS), the form zero-knowledge proof systems such as Groth16 start
//! from, checked by hand.
//!
//!     cargo run --example r1cs
//!
//! The statement: "I know x with x^3 + x + 5 = 35" (x = 3). Flattened into steps of one
//! multiplication each:
//!
//!     t1 = x * x
//!     t2 = t1 * x
//!     t3 = (t2 + x) * 1
//!     out = (t3 + 5) * 1
//!
//! With the witness w = (1, out, x, t1, t2, t3), every step is a row of three matrices A, B, C
//! with (A w)_i * (B w)_i = (C w)_i. A proof system checks all rows at once: with a random
//! challenge alpha, sum_i alpha^i * ((A w)_i (B w)_i - (C w)_i) = 0. If any row fails, this sum is
//! a non-zero polynomial in alpha, which is zero for at most 3 of the 2^255 possible alphas.

// ark-ff's MontConfig derive checks for an `asm` feature on fields of four or more limbs, which this
// crate doesn't define.
#![allow(unexpected_cfgs)]

use ark_ff::{AdditiveGroup, Fp256, MontBackend};
use field_matrix_utils::polynomial::evaluate_polynomial;
use field_matrix_utils::Matrix;

// The BLS12-381 scalar field, the one Groth16 proofs over BLS12-381 use.
#[derive(ark_ff::MontConfig)]
#[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[generator = "7"]
pub struct FrConfig;
type F = Fp256<MontBackend<FrConfig, 4>>;

fn matrix(rows: &[[u64; 6]]) -> Matrix<F> {
    Matrix::from_fn(rows.len(), 6, |i, j| F::from(rows[i][j]))
}

/// The row residuals (A w)_i * (B w)_i - (C w)_i: all zero for a valid witness.
fn residuals(a: &Matrix<F>, b: &Matrix<F>, c: &Matrix<F>, w: &[F]) -> Vec<F> {
    let (aw, bw, cw) = (
        a.mul_vec(w).unwrap(),
        b.mul_vec(w).unwrap(),
        c.mul_vec(w).unwrap(),
    );
    (0..aw.len()).map(|i| aw[i] * bw[i] - cw[i]).collect()
}

fn main() {
    // Columns:       1  out  x  t1  t2  t3
    let a = matrix(&[
        [0, 0, 1, 0, 0, 0], // x
        [0, 0, 0, 1, 0, 0], // t1
        [0, 0, 1, 0, 1, 0], // t2 + x
        [5, 0, 0, 0, 0, 1], // t3 + 5
    ]);
    let b = matrix(&[
        [0, 0, 1, 0, 0, 0], // x
        [0, 0, 1, 0, 0, 0], // x
        [1, 0, 0, 0, 0, 0], // 1
        [1, 0, 0, 0, 0, 0], // 1
    ]);
    let c = matrix(&[
        [0, 0, 0, 1, 0, 0], // t1
        [0, 0, 0, 0, 1, 0], // t2
        [0, 0, 0, 0, 0, 1], // t3
        [0, 1, 0, 0, 0, 0], // out
    ]);

    let witness: Vec<F> = [1, 35, 3, 9, 27, 30].map(F::from).to_vec();
    let r = residuals(&a, &b, &c, &witness);
    assert!(r.iter().all(|&x| x == F::ZERO));
    println!("witness (1, out, x, t1, t2, t3) = (1, 35, 3, 9, 27, 30): every row holds");

    // All rows at once: the residuals as the coefficients of a polynomial, evaluated at a random
    // challenge (fixed here; a verifier would pick it after seeing the commitment).
    let alpha = F::from(0x5eed_1234_abcd_u64);
    assert_eq!(evaluate_polynomial(&r, alpha), F::ZERO);
    println!("batched check at a random alpha: 0");

    // A wrong witness (x = 4 claims 4^3 + 4 + 5 = 35): rows fail, and the batched check sees it.
    let wrong: Vec<F> = [1, 35, 4, 16, 64, 68].map(F::from).to_vec();
    let r = residuals(&a, &b, &c, &wrong);
    let failing: Vec<usize> = (0..r.len()).filter(|&i| r[i] != F::ZERO).collect();
    println!("witness with x = 4: rows {failing:?} fail");
    assert_eq!(failing, vec![3]);
    assert_ne!(evaluate_polynomial(&r, alpha), F::ZERO);
    println!("batched check at the same alpha: not 0, rejected");
}
