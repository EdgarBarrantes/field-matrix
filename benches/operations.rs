//! Each operation computed from its definition (`naive`) and with the method on `Matrix`, on the
//! same matrices, over the BLS12-381 scalar field (255 bits, a common choice in zero-knowledge
//! proofs). `benches/run.sh` then `python3 benches/summarize.py > benches/RESULTS.md` writes the
//! comparison.
//!
//! Inputs are fixed pseudo-random matrices, so every run measures the same work.

// ark-ff's MontConfig derive checks for an `asm` feature on fields of four or more limbs, which this
// crate doesn't define.
#![allow(unexpected_cfgs)]

use std::hint::black_box;

use ark_ff::{Fp256, MontBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use field_matrix_utils::{naive, Matrix};

#[derive(ark_ff::MontConfig)]
#[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[generator = "7"]
pub struct FrConfig;
type F = Fp256<MontBackend<FrConfig, 4>>;

/// Small deterministic generator, so every run benchmarks the same matrices.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> F {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        F::from(self.0)
    }

    /// Random entries in a 255-bit field: invertible with overwhelming probability.
    fn matrix(&mut self, n: usize) -> Matrix<F> {
        Matrix::from_fn(n, n, |_, _| self.next())
    }
}

/// Benchmarks `naive` and `fast` on the same matrix for each size, in one group.
fn compare<R>(
    c: &mut Criterion,
    rng: &mut Rng,
    name: &str,
    sizes: &[usize],
    naive: impl Fn(&Matrix<F>, &[F]) -> R,
    fast: impl Fn(&Matrix<F>, &[F]) -> R,
) {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    for &n in sizes {
        let a = rng.matrix(n);
        let b: Vec<F> = (0..n).map(|_| rng.next()).collect();
        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bench, _| {
            bench.iter(|| black_box(naive(&a, &b)))
        });
        group.bench_with_input(BenchmarkId::new("fast", n), &n, |bench, _| {
            bench.iter(|| black_box(fast(&a, &b)))
        });
    }
    group.finish();
}

fn operations(c: &mut Criterion) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    // Sizes stay where the naive version finishes in well under a second.
    compare(
        c,
        &mut rng,
        "determinant_leibniz",
        &[4, 6, 8],
        |a, _| naive::determinant_leibniz(a),
        |a, _| a.determinant(),
    );
    compare(
        c,
        &mut rng,
        "determinant_laplace",
        &[4, 6, 8],
        |a, _| naive::determinant_laplace(a),
        |a, _| a.determinant(),
    );
    compare(
        c,
        &mut rng,
        "adjugate",
        &[4, 8, 16],
        |a, _| naive::adjugate(a),
        |a, _| a.adjugate(),
    );
    compare(
        c,
        &mut rng,
        "inverse",
        &[4, 8, 16],
        |a, _| naive::inverse(a),
        |a, _| a.inverse(),
    );
    compare(
        c,
        &mut rng,
        "solve",
        &[4, 16, 64],
        naive::solve_cramer,
        |a, b| a.solve(b),
    );
    compare(
        c,
        &mut rng,
        "pow",
        &[4, 16],
        |a, _| naive::pow(a, 64),
        |a, _| a.pow(64),
    );
}

criterion_group!(benches, operations);
criterion_main!(benches);
