//! Polynomials whose coefficients are matrices or vectors, evaluated at a point.
//!
//! Proof systems often batch many checks into one with a random linear combination: instead of
//! checking every `M_i`, they check `M_0 + alpha * M_1 + ... + alpha^k * M_k` for a random
//! challenge `alpha`. That is the polynomial `M_0 + M_1 x + ... + M_k x^k` evaluated at
//! `x = alpha`, which these functions compute with Horner's rule (`k` scalar multiplications).
//!
//! Also plain polynomials with field coefficients: [`evaluate_polynomial`], and [`interpolate`],
//! which finds the polynomial through given points by solving a system with the
//! [`Matrix::vandermonde`] matrix. Coefficients always go from the constant term up.

use crate::scalar::Scalar;
use crate::{Matrix, MatrixError};

/// Evaluates `M_0 + M_1 x + ... + M_k x^k` at `x = alpha`, where `coefficients` is
/// `[M_0, M_1, ..., M_k]`.
/// ## Example
/// ```
/// # use ark_ff::{Fp64, MontBackend};
/// # use field_matrix_utils::Matrix;
/// # #[derive(ark_ff::MontConfig)]
/// # #[modulus = "127"]
/// # #[generator = "6"]
/// # pub struct F127Config;
/// # type F = Fp64<MontBackend<F127Config, 1>>;
/// use field_matrix_utils::evaluate_matrix_polynomial;
///
/// let identity: Matrix<F> = Matrix::identity(2);
/// let ones: Matrix<F> = Matrix::from_fn(2, 2, |_, _| F::from(1));
///
/// // I + 3 * ones
/// let result = evaluate_matrix_polynomial(&[identity, ones], F::from(3)).unwrap();
/// assert_eq!(result, Matrix::from_rows(vec![vec![F::from(4), F::from(3)], vec![F::from(3), F::from(4)]]).unwrap());
/// ```
/// ## Errors
/// [`MatrixError::Empty`] if there are no coefficients; [`MatrixError::DimensionMismatch`] if the
/// matrices don't all have the same size.
pub fn evaluate_matrix_polynomial<F: Scalar>(
    coefficients: &[Matrix<F>],
    alpha: F,
) -> Result<Matrix<F>, MatrixError> {
    let (last, rest) = coefficients.split_last().ok_or(MatrixError::Empty)?;
    if rest
        .iter()
        .any(|c| (c.rows(), c.columns()) != (last.rows(), last.columns()))
    {
        return Err(MatrixError::DimensionMismatch);
    }
    Ok(rest
        .iter()
        .rev()
        .fold(last.clone(), |acc, c| &acc.scalar_mul(alpha) + c))
}

/// Evaluates `v_0 + v_1 x + ... + v_k x^k` at `x = alpha`, where `coefficients` is
/// `[v_0, v_1, ..., v_k]`.
/// ## Example
/// ```
/// # use ark_ff::{Fp64, MontBackend};
/// # #[derive(ark_ff::MontConfig)]
/// # #[modulus = "127"]
/// # #[generator = "6"]
/// # pub struct F127Config;
/// # type F = Fp64<MontBackend<F127Config, 1>>;
/// use field_matrix_utils::evaluate_vector_polynomial;
///
/// let v0 = vec![F::from(1), F::from(2)];
/// let v1 = vec![F::from(10), F::from(20)];
///
/// // v0 + 2 * v1
/// let result = evaluate_vector_polynomial(&[v0, v1], F::from(2)).unwrap();
/// assert_eq!(result, vec![F::from(21), F::from(42)]);
/// ```
/// ## Errors
/// [`MatrixError::Empty`] if there are no coefficients; [`MatrixError::DimensionMismatch`] if the
/// vectors don't all have the same length.
pub fn evaluate_vector_polynomial<F: Scalar>(
    coefficients: &[Vec<F>],
    alpha: F,
) -> Result<Vec<F>, MatrixError> {
    let (last, rest) = coefficients.split_last().ok_or(MatrixError::Empty)?;
    if rest.iter().any(|c| c.len() != last.len()) {
        return Err(MatrixError::DimensionMismatch);
    }
    Ok(rest.iter().rev().fold(last.clone(), |acc, c| {
        acc.iter().zip(c).map(|(&a, &b)| a * alpha + b).collect()
    }))
}

/// Evaluates `c_0 + c_1 x + ... + c_k x^k` at `x`, by Horner's rule; `coefficients` is
/// `[c_0, c_1, ..., c_k]` (an empty list is the zero polynomial).
/// ## Example
/// ```
/// # use ark_ff::{Fp64, MontBackend};
/// # #[derive(ark_ff::MontConfig)]
/// # #[modulus = "127"]
/// # #[generator = "6"]
/// # pub struct F127Config;
/// # type F = Fp64<MontBackend<F127Config, 1>>;
/// use field_matrix_utils::polynomial::evaluate_polynomial;
///
/// // 1 + 2x + 3x^2 at x = 2
/// let p = [F::from(1), F::from(2), F::from(3)];
/// assert_eq!(evaluate_polynomial(&p, F::from(2)), F::from(17));
/// ```
pub fn evaluate_polynomial<F: Scalar>(coefficients: &[F], x: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::ZERO_ELEMENT, |acc, &c| acc * x + c)
}

/// The polynomial of degree below `n` through the `n` points `(xs[i], ys[i])`, as coefficients from
/// the constant term up: the solution of `V c = ys` with `V` the [Vandermonde
/// matrix](Matrix::vandermonde) of `xs` (about n^3 multiplications; Lagrange or Newton
/// interpolation would take about n^2).
/// ## Example
/// ```
/// # use ark_ff::{Fp64, MontBackend};
/// # #[derive(ark_ff::MontConfig)]
/// # #[modulus = "127"]
/// # #[generator = "6"]
/// # pub struct F127Config;
/// # type F = Fp64<MontBackend<F127Config, 1>>;
/// use field_matrix_utils::polynomial::{evaluate_polynomial, interpolate};
///
/// // The line through (1, 3) and (2, 5): 1 + 2x
/// let p = interpolate(&[F::from(1), F::from(2)], &[F::from(3), F::from(5)]).unwrap();
/// assert_eq!(p, vec![F::from(1), F::from(2)]);
/// assert_eq!(evaluate_polynomial(&p, F::from(10)), F::from(21));
/// ```
/// ## Errors
/// [`MatrixError::DimensionMismatch`] if `xs` and `ys` have different lengths;
/// [`MatrixError::Singular`] if two points share the same `x`.
pub fn interpolate<F: Scalar>(xs: &[F], ys: &[F]) -> Result<Vec<F>, MatrixError> {
    if xs.len() != ys.len() {
        return Err(MatrixError::DimensionMismatch);
    }
    Matrix::vandermonde(xs).solve(ys)
}

impl<F: Scalar> Matrix<F> {
    /// The square Vandermonde matrix of `points`: row `i` is `1, x_i, x_i^2, ..., x_i^(n-1)`.
    /// Multiplying it by a polynomial's coefficients evaluates the polynomial at every point; it is
    /// invertible exactly when the points are distinct.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// let v: Matrix<F> = Matrix::vandermonde(&[F::from(2), F::from(3)]);
    /// assert_eq!(v, Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(1), F::from(3)]]).unwrap());
    /// ```
    pub fn vandermonde(points: &[F]) -> Matrix<F> {
        let n = points.len();
        let mut v = Matrix::zeros(n, n);
        for (i, &x) in points.iter().enumerate() {
            let mut power = F::ONE_ELEMENT;
            for j in 0..n {
                v[(i, j)] = power;
                power *= x;
            }
        }
        v
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{AdditiveGroup, Field, Fp64, MontBackend};

    #[derive(ark_ff::MontConfig)]
    #[modulus = "127"]
    #[generator = "6"]
    pub struct F127Config;
    type F = Fp64<MontBackend<F127Config, 1>>;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> F {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            F::from(self.0 % 1000)
        }
    }

    fn random_matrix(rng: &mut Rng, rows: usize, columns: usize) -> Matrix<F> {
        Matrix::from_fn(rows, columns, |_, _| rng.next())
    }

    #[test]
    fn matrices_match_the_sum_of_powers() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for k in 1..6 {
            let coefficients: Vec<Matrix<F>> =
                (0..k).map(|_| random_matrix(&mut rng, 2, 3)).collect();
            let alpha = rng.next();

            let mut expected = Matrix::zeros(2, 3);
            let mut power = F::ONE;
            for c in &coefficients {
                expected = &expected + &c.scalar_mul(power);
                power *= alpha;
            }

            assert_eq!(
                evaluate_matrix_polynomial(&coefficients, alpha),
                Ok(expected)
            );
        }
    }

    #[test]
    fn vectors_match_the_sum_of_powers() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for k in 1..6 {
            let coefficients: Vec<Vec<F>> = (0..k)
                .map(|_| (0..4).map(|_| rng.next()).collect())
                .collect();
            let alpha = rng.next();

            let mut expected = vec![F::ZERO; 4];
            let mut power = F::ONE;
            for c in &coefficients {
                for (e, &x) in expected.iter_mut().zip(c) {
                    *e += x * power;
                }
                power *= alpha;
            }

            assert_eq!(
                evaluate_vector_polynomial(&coefficients, alpha),
                Ok(expected)
            );
        }
    }

    #[test]
    fn special_points() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let c: Vec<Matrix<F>> = (0..3).map(|_| random_matrix(&mut rng, 2, 2)).collect();

        // at 0 only the constant term is left; at 1 it is the plain sum
        assert_eq!(evaluate_matrix_polynomial(&c, F::ZERO), Ok(c[0].clone()));
        assert_eq!(
            evaluate_matrix_polynomial(&c, F::ONE),
            Ok(&(&c[0] + &c[1]) + &c[2])
        );
    }

    #[test]
    fn empty_and_mismatched() {
        assert_eq!(
            evaluate_matrix_polynomial::<F>(&[], F::ONE),
            Err(MatrixError::Empty)
        );
        assert_eq!(
            evaluate_vector_polynomial::<F>(&[], F::ONE),
            Err(MatrixError::Empty)
        );
        assert_eq!(
            evaluate_vector_polynomial(&[vec![F::ONE], vec![F::ONE, F::ONE]], F::ONE),
            Err(MatrixError::DimensionMismatch)
        );
        let mut rng = Rng(7);
        let small = random_matrix(&mut rng, 2, 2);
        let tall = random_matrix(&mut rng, 3, 2);
        assert_eq!(
            evaluate_matrix_polynomial(&[small, tall], F::ONE),
            Err(MatrixError::DimensionMismatch)
        );
    }
}
