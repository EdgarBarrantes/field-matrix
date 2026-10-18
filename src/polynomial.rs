//! Polynomials whose coefficients are matrices or vectors, evaluated at a point.
//!
//! Proof systems often batch many checks into one with a random linear combination: instead of
//! checking every `M_i`, they check `M_0 + alpha * M_1 + ... + alpha^k * M_k` for a random
//! challenge `alpha`. That is the polynomial `M_0 + M_1 x + ... + M_k x^k` evaluated at
//! `x = alpha`, which these functions compute with Horner's rule (`k` scalar multiplications).

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
