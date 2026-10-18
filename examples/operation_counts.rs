//! How many field operations each version of an operation needs, by matrix size.
//!
//!     cargo run --example operation_counts
//!
//! Prints a Markdown table: multiplications, and in brackets inversions (each inversion costs
//! about as much as a few dozen multiplications). Entries are pseudo-random in the 255-bit
//! BLS12-381 scalar field, so every matrix is invertible and no pivot is zero by luck.

// ark-ff's MontConfig derive checks for an `asm` feature on fields of four or more limbs, which this
// crate doesn't define.
#![allow(unexpected_cfgs)]

use ark_ff::{Fp256, MontBackend};
use field_matrix_utils::counting::{self, Counts};
use field_matrix_utils::{naive, Matrix};

#[derive(ark_ff::MontConfig)]
#[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[generator = "7"]
pub struct FrConfig;
type F = Fp256<MontBackend<FrConfig, 4>>;
/// Entries that count the operations done with them.
type C = counting::Counting<F>;

const SIZES: [usize; 6] = [2, 3, 4, 5, 6, 7];

fn cell(counts: Counts) -> String {
    format!("{} ({})", counts.multiplications, counts.inversions)
}

fn main() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        F::from(state)
    };

    type Row = (&'static str, fn(&Matrix<C>, &[C]));
    let rows: [Row; 8] = [
        ("determinant (elimination)", |a, _| {
            let _ = a.determinant();
        }),
        ("naive::determinant_leibniz", |a, _| {
            let _ = naive::determinant_leibniz(a);
        }),
        ("naive::determinant_laplace", |a, _| {
            let _ = naive::determinant_laplace(a);
        }),
        ("adjugate (det * inverse)", |a, _| {
            let _ = a.adjugate();
        }),
        ("naive::adjugate (cofactors)", |a, _| {
            let _ = naive::adjugate(a);
        }),
        ("inverse (Gauss-Jordan)", |a, _| {
            let _ = a.inverse();
        }),
        ("naive::inverse (adjugate / det)", |a, _| {
            let _ = naive::inverse(a);
        }),
        ("naive::solve_cramer (Cramer's rule)", |a, b| {
            let _ = naive::solve_cramer(a, b);
        }),
    ];

    let matrices: Vec<(Matrix<C>, Vec<C>)> = SIZES
        .iter()
        .map(|&n| {
            let a = Matrix::from_fn(n, n, |_, _| next());
            let b: Vec<F> = (0..n).map(|_| next()).collect();
            (
                counting::wrap(&a),
                b.into_iter().map(counting::Counting).collect(),
            )
        })
        .collect();

    println!("Field multiplications (inversions in brackets), by matrix size n:\n");
    println!(
        "| Operation | {} |",
        SIZES.map(|n| format!("n = {n}")).join(" | ")
    );
    println!("|---|{}", "---|".repeat(SIZES.len()));
    for (name, run) in rows {
        let cells: Vec<String> = matrices
            .iter()
            .map(|(a, b)| cell(counting::count(|| run(a, b)).1))
            .collect();
        println!("| {name} | {} |", cells.join(" | "));
    }
    let solve: Vec<String> = matrices
        .iter()
        .map(|(a, b)| {
            cell(
                counting::count(|| {
                    let _ = a.solve(b);
                })
                .1,
            )
        })
        .collect();
    println!("| solve (Gauss-Jordan) | {} |", solve.join(" | "));
}
