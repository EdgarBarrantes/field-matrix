//! Benchmarks of the main operations, over the BLS12-381 scalar field (255 bits, a common choice in
//! zero-knowledge proofs). Run with `cargo bench`; see `benches/RESULTS.md` for recorded results.
//!
//! Inputs are fixed pseudo-random matrices, so every run measures the same work. Each operation
//! gets its own copy of the input outside the timed part (`iter_batched`), so only the operation
//! is measured.

// ark-ff's MontConfig derive checks for an `asm` feature on fields of four or more limbs, which this
// crate doesn't define.
#![allow(unexpected_cfgs)]

use std::hint::black_box;

use ark_ff::{Fp256, MontBackend};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use field_matrix_utils::{evaluate_matrix_polynomial, Matrix};

#[derive(ark_ff::MontConfig)]
#[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[generator = "7"]
pub struct FrConfig;
type F = Fp256<MontBackend<FrConfig, 4>>;

/// Matrix sizes (n x n) for most operations.
const SIZES: [usize; 3] = [4, 16, 64];
/// The adjugate by cofactors costs about n^5: n = 64 would take minutes per iteration.
const ADJOINT_SIZES: [usize; 3] = [4, 8, 16];
/// Number of coefficients for the polynomial evaluation.
const POLYNOMIAL_TERMS: usize = 8;

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
        Matrix::new(
            (0..n)
                .map(|_| (0..n).map(|_| self.next()).collect())
                .collect(),
        )
    }
}

fn operations(c: &mut Criterion) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    let mut group = c.benchmark_group("multiply");
    for n in SIZES {
        let (a, b) = (rng.matrix(n), rng.matrix(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(a * b),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    let mut group = c.benchmark_group("determinant");
    for n in SIZES {
        let a = rng.matrix(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter_batched(
                || a.clone(),
                |a| black_box(a.determinant()),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    let mut group = c.benchmark_group("inverse");
    for n in SIZES {
        let a = rng.matrix(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter_batched(
                || a.clone(),
                |a| black_box(a.inverse()),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    let mut group = c.benchmark_group("solve");
    for n in SIZES {
        let a = rng.matrix(n);
        let b: Vec<F> = (0..n).map(|_| rng.next()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(a.ax_b_solve_for_x(b)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    let mut group = c.benchmark_group("adjoint");
    group.sample_size(10);
    for n in ADJOINT_SIZES {
        let a = rng.matrix(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter_batched(
                || a.clone(),
                |a| black_box(a.adjoint()),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    let mut group = c.benchmark_group("evaluate_matrix_polynomial");
    for n in SIZES {
        let coefficients: Vec<Matrix<F>> = (0..POLYNOMIAL_TERMS).map(|_| rng.matrix(n)).collect();
        let alpha = rng.next();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(evaluate_matrix_polynomial(&coefficients, alpha)))
        });
    }
    group.finish();
}

criterion_group!(benches, operations);
criterion_main!(benches);
