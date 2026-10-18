//! Counting field operations: how much work an algorithm does, independent of the machine.
//!
//! [`Counting<F>`] wraps a field element and counts every addition (and subtraction),
//! multiplication and inversion done with it. Wrap a matrix's entries, run an operation inside
//! [`count`], and read the totals:
//!
//! ```
//! # use ark_ff::{Fp64, MontBackend};
//! # #[derive(ark_ff::MontConfig)]
//! # #[modulus = "127"]
//! # #[generator = "6"]
//! # pub struct F127Config;
//! # type F = Fp64<MontBackend<F127Config, 1>>;
//! use ark_ff::Field;
//! use field_matrix_utils::counting;
//! use field_matrix_utils::{naive, Matrix};
//!
//! // A Vandermonde matrix with distinct nodes 2..=7: always invertible.
//! let a: Matrix<F> = Matrix::from_fn(6, 6, |i, j| F::from(i as u64 + 2).pow([j as u64]));
//! let a = counting::wrap(&a);
//!
//! let (fast, fast_counts) = counting::count(|| a.adjugate());
//! let (slow, slow_counts) = counting::count(|| naive::adjugate(&a));
//! assert_eq!(fast, slow);
//! assert!(fast_counts.multiplications < slow_counts.multiplications);
//! assert!(fast_counts.inversions * 10 < slow_counts.inversions);
//! ```
//!
//! Counts are kept per thread. Negations and comparisons are not counted.

use core::cell::Cell;
use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_ff::Field;

use crate::scalar::Scalar;
use crate::Matrix;

/// A field element that counts the operations done with it (see the [module docs](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Counting<F>(pub F);

/// Operation totals from [`count`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Additions and subtractions.
    pub additions: u64,
    /// Multiplications.
    pub multiplications: u64,
    /// Inversions (each one costs about as much as a few dozen multiplications).
    pub inversions: u64,
}

impl Add for Counts {
    type Output = Counts;

    fn add(self, other: Counts) -> Counts {
        Counts {
            additions: self.additions + other.additions,
            multiplications: self.multiplications + other.multiplications,
            inversions: self.inversions + other.inversions,
        }
    }
}

thread_local! {
    static COUNTS: Cell<Counts> = const { Cell::new(Counts { additions: 0, multiplications: 0, inversions: 0 }) };
}

fn record(change: impl FnOnce(&mut Counts)) {
    COUNTS.with(|counts| {
        let mut c = counts.get();
        change(&mut c);
        counts.set(c);
    });
}

/// Runs `f` and returns its result with the operations [`Counting`] values did meanwhile.
/// Calls can be nested: an outer count includes the inner ones.
pub fn count<R>(f: impl FnOnce() -> R) -> (R, Counts) {
    let outer = COUNTS.with(|counts| counts.replace(Counts::default()));
    let result = f();
    let inner = COUNTS.with(|counts| counts.replace(Counts::default()));
    COUNTS.with(|counts| counts.set(outer + inner));
    (result, inner)
}

/// The same matrix with [`Counting`] entries.
pub fn wrap<F: Field>(a: &Matrix<F>) -> Matrix<Counting<F>> {
    Matrix::from_fn(a.rows(), a.columns(), |i, j| Counting(a[(i, j)]))
}

/// The same matrix with plain entries again.
pub fn unwrap<F: Field>(a: &Matrix<Counting<F>>) -> Matrix<F> {
    Matrix::from_fn(a.rows(), a.columns(), |i, j| a[(i, j)].0)
}

impl<F: fmt::Display> fmt::Display for Counting<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<F: Field> Scalar for Counting<F> {
    const ZERO_ELEMENT: Self = Counting(<F as Scalar>::ZERO_ELEMENT);
    const ONE_ELEMENT: Self = Counting(<F as Scalar>::ONE_ELEMENT);

    fn invert(&self) -> Option<Self> {
        record(|c| c.inversions += 1);
        self.0.inverse().map(Counting)
    }
}

impl<F: Field> Add for Counting<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        record(|c| c.additions += 1);
        Counting(self.0 + other.0)
    }
}

// The `+= 1` counts the operation; clippy takes it for a mistake in the arithmetic.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Field> Sub for Counting<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        record(|c| c.additions += 1);
        Counting(self.0 - other.0)
    }
}

// The `+= 1` counts the operation; clippy takes it for a mistake in the arithmetic.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: Field> Mul for Counting<F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        record(|c| c.multiplications += 1);
        Counting(self.0 * other.0)
    }
}

impl<F: Field> Neg for Counting<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Counting(-self.0)
    }
}

impl<F: Field> AddAssign for Counting<F> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<F: Field> SubAssign for Counting<F> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<F: Field> MulAssign for Counting<F> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Fp64, MontBackend};

    #[derive(ark_ff::MontConfig)]
    #[modulus = "127"]
    #[generator = "6"]
    pub struct F127Config;
    type F = Fp64<MontBackend<F127Config, 1>>;

    #[test]
    fn counts_each_kind() {
        let (x, counts) = count(|| {
            let a = Counting(F::from(3u64));
            let b = Counting(F::from(5u64));
            let mut c = a * b + a - b;
            c *= a;
            c += b;
            c.invert()
        });
        assert_eq!(
            x,
            Some(Counting(
                (F::from(15u64 + 3 - 5) * F::from(3u64) + F::from(5u64))
                    .inverse()
                    .unwrap()
            ))
        );
        assert_eq!(
            counts,
            Counts {
                additions: 3,
                multiplications: 2,
                inversions: 1
            }
        );
    }

    #[test]
    fn nested_counts_add_up() {
        let one = Counting(F::from(1u64));
        let (inner, outer) = count(|| {
            let _ = one + one;
            let (_, inner) = count(|| one * one);
            inner
        });
        assert_eq!(
            inner,
            Counts {
                multiplications: 1,
                ..Counts::default()
            }
        );
        assert_eq!(
            outer,
            Counts {
                additions: 1,
                multiplications: 1,
                ..Counts::default()
            }
        );
    }

    #[test]
    fn schoolbook_multiplication_costs_n_cubed() {
        let a = wrap(&Matrix::<F>::from_fn(5, 5, |i, j| {
            F::from((i + 2 * j + 1) as u64)
        }));
        let (product, counts) = count(|| &a * &a);
        assert_eq!(counts.multiplications, 125);
        assert_eq!(counts.additions, 125);
        assert_eq!(unwrap(&product), &unwrap(&a) * &unwrap(&a));
    }
}
