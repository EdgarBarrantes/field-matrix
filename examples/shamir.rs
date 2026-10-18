//! Shamir secret sharing: split a secret into 5 shares so that any 3 rebuild it and 2 reveal
//! nothing.
//!
//!     cargo run --example shamir
//!
//! The secret is the constant term of a random polynomial of degree 2; each share is one point
//! on it. Three points fix a polynomial of degree 2 (interpolation, a Vandermonde system); two
//! points leave a whole line of polynomials, one for every possible secret.

// ark-ff's MontConfig derive checks for an `asm` feature on fields of four or more limbs, which this
// crate doesn't define.
#![allow(unexpected_cfgs)]

use ark_ff::{AdditiveGroup, Fp256, MontBackend};
use field_matrix_utils::polynomial::{evaluate_polynomial, interpolate};
use field_matrix_utils::Matrix;

// The BLS12-381 scalar field, common in zero-knowledge proofs.
#[derive(ark_ff::MontConfig)]
#[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[generator = "7"]
pub struct FrConfig;
type F = Fp256<MontBackend<FrConfig, 4>>;

const THRESHOLD: usize = 3;
const SHARES: u64 = 5;

fn main() {
    let secret = F::from(424242u64);

    // A polynomial of degree THRESHOLD - 1 with the secret as constant term. The other
    // coefficients must be random and kept secret (fixed here so the example always prints the
    // same; use a cryptographic random generator in real use).
    let polynomial = [secret, F::from(987_654_321u64), F::from(123_456_789u64)];

    // Share i is the point (i, p(i)), for i = 1..=5 (never 0: p(0) is the secret).
    let shares: Vec<(F, F)> = (1..=SHARES)
        .map(|i| (F::from(i), evaluate_polynomial(&polynomial, F::from(i))))
        .collect();
    println!("{SHARES} shares, any {THRESHOLD} of them rebuild the secret.");

    // Any 3 shares: interpolate the polynomial through them and read its constant term.
    for chosen in [[0, 1, 2], [0, 2, 4], [1, 3, 4]] {
        let xs: Vec<F> = chosen.iter().map(|&k| shares[k].0).collect();
        let ys: Vec<F> = chosen.iter().map(|&k| shares[k].1).collect();
        let rebuilt = interpolate(&xs, &ys).expect("distinct points");
        println!("shares {chosen:?}: secret = {}", rebuilt[0]);
        assert_eq!(rebuilt[0], secret);
    }

    // Only 2 shares: the coefficients c with V c = ys, where V has a row (1, x, x^2) per share,
    // are a whole line, solution + t * kernel. The kernel vector has a non-zero constant term, so
    // every secret is possible: 2 shares say nothing about it.
    let known = &shares[..2];
    let v = Matrix::from_fn(known.len(), THRESHOLD, |i, j| {
        (0..j).fold(F::from(1u64), |power, _| power * known[i].0)
    });
    let ys: Vec<F> = known.iter().map(|share| share.1).collect();
    let solutions = v.solve_all(&ys).unwrap().expect("there are solutions");
    assert_eq!(solutions.kernel.len(), 1);
    let direction = &solutions.kernel[0];
    assert_ne!(direction[0], F::ZERO);
    println!(
        "2 shares: the polynomials through them form a line, and their constant terms take every \
         value: the secret could be anything."
    );
}
