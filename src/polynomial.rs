//! Polynomials whose coefficients are matrices or vectors, evaluated at a point.
//!
//! Proof systems often batch many checks into one with a random linear combination: instead of
//! checking every `M_i`, they check `M_0 + alpha * M_1 + ... + alpha^k * M_k` for a random
//! challenge `alpha`. That is the polynomial `M_0 + M_1 x + ... + M_k x^k` evaluated at
//! `x = alpha`, which these functions compute with Horner's rule (`k` scalar multiplications).

use ark_ff::Field;

use crate::Matrix;

/// Evaluates `M_0 + M_1 x + ... + M_k x^k` at `x = alpha`, where `coefficients` is
/// `[M_0, M_1, ..., M_k]`.
///
/// Returns `None` if `coefficients` is empty.
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
/// let identity: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(0)], vec![F::from(0), F::from(1)]]);
/// let ones: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(1)], vec![F::from(1), F::from(1)]]);
///
/// // I + 3 * ones
/// let result = evaluate_matrix_polynomial(&[identity, ones], F::from(3)).unwrap();
/// assert_eq!(
///     result,
///     Matrix::new(vec![vec![F::from(4), F::from(3)], vec![F::from(3), F::from(4)]])
/// );
/// ```
/// ## Panics
/// Panics if the matrices don't all have the same size.
pub fn evaluate_matrix_polynomial<F: Field>(
    coefficients: &[Matrix<F>],
    alpha: F,
) -> Option<Matrix<F>> {
    let (last, rest) = coefficients.split_last()?;
    Some(
        rest.iter()
            .rev()
            .fold(last.clone(), |acc, c| acc.scalar_mul(alpha) + c.clone()),
    )
}

/// Evaluates `v_0 + v_1 x + ... + v_k x^k` at `x = alpha`, where `coefficients` is
/// `[v_0, v_1, ..., v_k]`.
///
/// Returns `None` if `coefficients` is empty.
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
/// ## Panics
/// Panics if the vectors don't all have the same length.
pub fn evaluate_vector_polynomial<F: Field>(coefficients: &[Vec<F>], alpha: F) -> Option<Vec<F>> {
    let (last, rest) = coefficients.split_last()?;
    Some(rest.iter().rev().fold(last.clone(), |acc, c| {
        assert_eq!(acc.len(), c.len(), "Vectors have different lengths");
        acc.iter().zip(c).map(|(&a, &b)| a * alpha + b).collect()
    }))
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
        Matrix::new(
            (0..rows)
                .map(|_| (0..columns).map(|_| rng.next()).collect())
                .collect(),
        )
    }

    #[test]
    fn matrices_match_the_sum_of_powers() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for k in 1..6 {
            let coefficients: Vec<Matrix<F>> =
                (0..k).map(|_| random_matrix(&mut rng, 2, 3)).collect();
            let alpha = rng.next();

            let mut expected = coefficients[0].clone().scalar_mul(F::ZERO);
            let mut power = F::ONE;
            for c in &coefficients {
                expected = expected + c.clone().scalar_mul(power);
                power *= alpha;
            }

            assert_eq!(
                evaluate_matrix_polynomial(&coefficients, alpha),
                Some(expected)
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
                Some(expected)
            );
        }
    }

    #[test]
    fn special_points() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let coefficients: Vec<Matrix<F>> = (0..3).map(|_| random_matrix(&mut rng, 2, 2)).collect();

        // at 0 only the constant term is left; at 1 it is the plain sum
        assert_eq!(
            evaluate_matrix_polynomial(&coefficients, F::ZERO),
            Some(coefficients[0].clone())
        );
        assert_eq!(
            evaluate_matrix_polynomial(&coefficients, F::ONE),
            Some(coefficients[0].clone() + coefficients[1].clone() + coefficients[2].clone())
        );
    }

    #[test]
    fn empty_is_none() {
        assert_eq!(evaluate_matrix_polynomial::<F>(&[], F::ONE), None);
        assert_eq!(evaluate_vector_polynomial::<F>(&[], F::ONE), None);
    }

    #[test]
    #[should_panic(expected = "Vectors have different lengths")]
    fn vectors_of_different_lengths() {
        evaluate_vector_polynomial(&[vec![F::ONE], vec![F::ONE, F::ONE]], F::ONE);
    }

    #[test]
    #[should_panic(expected = "different number of rows")]
    fn matrices_of_different_sizes() {
        let mut rng = Rng(7);
        let small = random_matrix(&mut rng, 2, 2);
        let big = random_matrix(&mut rng, 3, 2);
        evaluate_matrix_polynomial(&[small, big], F::ONE);
    }
}
