//! Reed-Solomon erasure coding: store 4 data symbols as 7, survive the loss of any 3, and notice a
//! corrupted one.
//!
//!     cargo run --example reed_solomon
//!
//! The data are the coefficients of a polynomial of degree 3; the stored symbols are its values at
//! 7 points (a 7 x 4 Vandermonde matrix times the data). Any 4 values fix the polynomial again.
//! With more than 4 values, they must all lie on one polynomial of degree 3: if one was changed,
//! the overdetermined system has no solution.

use ark_ff::{Fp64, MontBackend};
use field_matrix_utils::Matrix;

// GF(257): every byte value 0..=255 is a field element.
#[derive(ark_ff::MontConfig)]
#[modulus = "257"]
#[generator = "3"]
pub struct F257Config;
type F = Fp64<MontBackend<F257Config, 1>>;

const DATA: usize = 4;
const STORED: usize = 7;

/// The rows of the evaluation matrix for the given points: row (1, x, x^2, x^3).
fn evaluation_matrix(points: &[u64]) -> Matrix<F> {
    Matrix::from_fn(points.len(), DATA, |i, j| F::from(points[i].pow(j as u32)))
}

fn main() {
    let data: Vec<F> = b"RUST".iter().map(|&byte| F::from(byte as u64)).collect();
    let points: Vec<u64> = (1..=STORED as u64).collect();

    // Encode: 7 values of the polynomial whose coefficients are the data.
    let stored = evaluation_matrix(&points).mul_vec(&data).unwrap();
    println!("data {:?} stored as {} symbols", "RUST", stored.len());

    // Lose 3 of them: the 4 left fix the polynomial. Solve the 4 x 4 system for the data.
    let kept = [0, 2, 5, 6];
    let kept_points: Vec<u64> = kept.iter().map(|&k| points[k]).collect();
    let kept_values: Vec<F> = kept.iter().map(|&k| stored[k]).collect();
    let recovered = evaluation_matrix(&kept_points)
        .solve(&kept_values)
        .expect("distinct points: the Vandermonde matrix is invertible");
    let text: String = recovered
        .iter()
        .map(|x| x.to_string().parse::<u8>().unwrap() as char)
        .collect();
    println!("symbols 1, 3 and 4 lost; recovered from the other 4: {text:?}");
    assert_eq!(recovered, data);

    // Corrupt one symbol, keep all 7: the system "these 7 values lie on one polynomial of degree 3"
    // has no solution any more, so the damage shows.
    let all = evaluation_matrix(&points);
    assert!(all.solve_all(&stored).unwrap().is_some());
    let mut damaged = stored.clone();
    damaged[3] += F::from(1u64);
    assert_eq!(all.solve_all(&damaged), Ok(None));
    println!("one symbol changed: the 7 values no longer fit a polynomial of degree 3, detected.");
}
