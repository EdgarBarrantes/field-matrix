//! Operations computed straight from their definitions, to read, learn from and compare with the
//! [`Matrix`] methods, which compute the same results with less work.
//!
//! The tests check that each one agrees with its counterpart. To see the difference in work, count
//! the operations with [`counting`](crate::counting); to see it in time, `cargo bench`.
//!
//! | Function | Cost (field multiplications) | Fast counterpart | Cost |
//! |---|---|---|---|
//! | [`determinant_leibniz`] | about n * n! | [`Matrix::determinant`] | about n^3 / 3 |
//! | [`determinant_laplace`] | about e * n! | [`Matrix::determinant`] | about n^3 / 3 |
//! | [`adjugate`] | about n^5 / 3 | [`Matrix::adjugate`] | about n^3 |
//! | [`inverse`] | about n^5 / 3 | [`Matrix::inverse`] | about n^3 |
//! | [`solve_cramer`] | about n^4 / 3 | [`Matrix::solve`] | about n^3 |
//! | [`pow`] | k * n^3 | [`Matrix::pow`] | up to 2 * log2(k) * n^3 |

use crate::scalar::Scalar;
use crate::{Matrix, MatrixError};

/// The determinant by the Leibniz formula: the sum over all `n!` permutations `s` of
/// `sign(s) * a[0][s(0)] * ... * a[n-1][s(n-1)]`.
///
/// Permutations are visited in Heap's order, where each one differs from the last by one swap, so
/// the sign just flips each time.
/// ## Errors
/// [`MatrixError::NotSquare`].
pub fn determinant_leibniz<F: Scalar>(a: &Matrix<F>) -> Result<F, MatrixError> {
    if !a.is_square() {
        return Err(MatrixError::NotSquare);
    }
    let n = a.rows();
    let mut permutation: Vec<usize> = (0..n).collect();
    let product = |p: &[usize]| (0..n).fold(F::ONE_ELEMENT, |acc, i| acc * a[(i, p[i])]);

    let mut total = product(&permutation);
    let mut positive = true;
    // Heap's algorithm, iterative.
    let mut counters = vec![0; n];
    let mut i = 1;
    while i < n {
        if counters[i] < i {
            let j = if i % 2 == 0 { 0 } else { counters[i] };
            permutation.swap(j, i);
            positive = !positive;
            let term = product(&permutation);
            total = if positive { total + term } else { total - term };
            counters[i] += 1;
            i = 1;
        } else {
            counters[i] = 0;
            i += 1;
        }
    }
    Ok(total)
}

/// The determinant by cofactor (Laplace) expansion along the first row:
/// `det(A) = sum over j of (-1)^j * a[0][j] * det(minor(0, j))`, recursively.
/// ## Errors
/// [`MatrixError::NotSquare`].
pub fn determinant_laplace<F: Scalar>(a: &Matrix<F>) -> Result<F, MatrixError> {
    if !a.is_square() {
        return Err(MatrixError::NotSquare);
    }
    Ok(laplace(a))
}

fn laplace<F: Scalar>(a: &Matrix<F>) -> F {
    match a.rows() {
        0 => F::ONE_ELEMENT,
        1 => a[(0, 0)],
        n => (0..n).fold(F::ZERO_ELEMENT, |total, j| {
            let term = a[(0, j)] * laplace(&a.minor(0, j));
            if j % 2 == 0 {
                total + term
            } else {
                total - term
            }
        }),
    }
}

/// The adjugate from its definition: the transpose of the matrix of cofactors, where the cofactor
/// of `(i, j)` is `(-1)^(i + j)` times the determinant of the matrix without row `i` and column
/// `j`. n^2 determinants of size n - 1.
/// ## Errors
/// [`MatrixError::NotSquare`].
pub fn adjugate<F: Scalar>(a: &Matrix<F>) -> Result<Matrix<F>, MatrixError> {
    if !a.is_square() {
        return Err(MatrixError::NotSquare);
    }
    let n = a.rows();
    if n == 1 {
        return Ok(Matrix::identity(1));
    }
    let mut adjugate = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let minor = a.minor(i, j).determinant()?;
            // Transposed: the cofactor of (i, j) goes to (j, i).
            adjugate[(j, i)] = if (i + j) % 2 == 0 { minor } else { -minor };
        }
    }
    Ok(adjugate)
}

/// The inverse as `adj(A) / det(A)`.
/// ## Errors
/// [`MatrixError::NotSquare`], [`MatrixError::Singular`].
pub fn inverse<F: Scalar>(a: &Matrix<F>) -> Result<Matrix<F>, MatrixError> {
    let det_inverse = a.determinant()?.invert().ok_or(MatrixError::Singular)?;
    Ok(adjugate(a)?.scalar_mul(det_inverse))
}

/// Solves `A x = b` by Cramer's rule: `x_i = det(A_i) / det(A)`, where `A_i` is `A` with column
/// `i` replaced by `b`. n + 1 determinants (each by elimination).
/// ## Errors
/// [`MatrixError::NotSquare`], [`MatrixError::DimensionMismatch`] if `b` doesn't have one entry
/// per row, [`MatrixError::Singular`].
pub fn solve_cramer<F: Scalar>(a: &Matrix<F>, b: &[F]) -> Result<Vec<F>, MatrixError> {
    if !a.is_square() {
        return Err(MatrixError::NotSquare);
    }
    if b.len() != a.rows() {
        return Err(MatrixError::DimensionMismatch);
    }
    let det_inverse = a.determinant()?.invert().ok_or(MatrixError::Singular)?;
    let n = a.rows();
    (0..n)
        .map(|i| {
            let replaced = Matrix::from_fn(n, n, |r, c| if c == i { b[r] } else { a[(r, c)] });
            Ok(replaced.determinant()? * det_inverse)
        })
        .collect()
}

/// The matrix to the power `k` by multiplying it in `k` times, starting from the identity.
/// ## Errors
/// [`MatrixError::NotSquare`].
pub fn pow<F: Scalar>(a: &Matrix<F>, k: u64) -> Result<Matrix<F>, MatrixError> {
    if !a.is_square() {
        return Err(MatrixError::NotSquare);
    }
    let mut result = Matrix::identity(a.rows());
    for _ in 0..k {
        result = &result * a;
    }
    Ok(result)
}
