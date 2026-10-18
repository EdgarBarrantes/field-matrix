//! Dense matrices over a finite field: [`Matrix`], and the [`MatrixError`] its methods return.

use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};
use std::fmt;

use crate::scalar::Scalar;

/// A dense `rows x columns` matrix over the field `F`, stored row by row in one `Vec`.
///
/// Arithmetic operators (`+`, `-`, `*`, unary `-`) work on owned matrices and on references
/// (`&a + &b`) and panic if the sizes don't fit. Methods that can fail return a [`MatrixError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<F: Scalar> {
    rows: usize,
    columns: usize,
    /// Row-major: the entry at `(i, j)` is `data[i * columns + j]`.
    data: Vec<F>,
}

/// Why a matrix operation couldn't be done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatrixError {
    /// The data doesn't describe a matrix: rows of different lengths, or `rows * columns` entries
    /// not matching the data.
    InvalidShape,
    /// The sizes of the operands don't fit together.
    DimensionMismatch,
    /// The operation needs a square matrix.
    NotSquare,
    /// The matrix is not invertible (its determinant is zero).
    Singular,
    /// LU decomposition without row exchanges met a zero pivot, as in `[[0, 1], [1, 0]]`.
    ZeroPivot,
    /// A row or column index outside the matrix.
    OutOfBounds,
    /// An empty list where at least one element is needed.
    Empty,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MatrixError::InvalidShape => "the data doesn't have the shape of a matrix",
            MatrixError::DimensionMismatch => "the sizes of the operands don't fit together",
            MatrixError::NotSquare => "the matrix is not square",
            MatrixError::Singular => "the matrix is not invertible (determinant zero)",
            MatrixError::ZeroPivot => "zero pivot: no LU decomposition without row exchanges",
            MatrixError::OutOfBounds => "index outside the matrix",
            MatrixError::Empty => "an empty list where at least one element is needed",
        })
    }
}

impl std::error::Error for MatrixError {}

/// Prints one row per line, entries as numbers, aligned by column:
///
/// ```text
/// [ 1, 126]
/// [30,   4]
/// ```
impl<F: Scalar> fmt::Display for Matrix<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cells: Vec<String> = self.data.iter().map(|x| x.to_string()).collect();
        let widths: Vec<usize> = (0..self.columns)
            .map(|j| {
                (0..self.rows)
                    .map(|i| cells[i * self.columns + j].len())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        for i in 0..self.rows {
            let row: Vec<String> = (0..self.columns)
                .map(|j| format!("{:>width$}", cells[i * self.columns + j], width = widths[j]))
                .collect();
            writeln!(f, "[{}]", row.join(", "))?;
        }
        Ok(())
    }
}

impl<F: Scalar> Index<(usize, usize)> for Matrix<F> {
    type Output = F;

    /// The entry at row `i`, column `j`. Panics if it is outside the matrix; [`Matrix::get`]
    /// returns an `Option` instead.
    fn index(&self, (i, j): (usize, usize)) -> &F {
        assert!(i < self.rows && j < self.columns, "Index out of bounds");
        &self.data[i * self.columns + j]
    }
}

impl<F: Scalar> IndexMut<(usize, usize)> for Matrix<F> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut F {
        assert!(i < self.rows && j < self.columns, "Index out of bounds");
        &mut self.data[i * self.columns + j]
    }
}

impl<F: Scalar> Add<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    fn add(self, other: &Matrix<F>) -> Matrix<F> {
        self.assert_same_size(other);
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a + b)
            .collect();
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }
}

impl<F: Scalar> Sub<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    fn sub(self, other: &Matrix<F>) -> Matrix<F> {
        self.assert_same_size(other);
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a - b)
            .collect();
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }
}

impl<F: Scalar> Mul<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    /// Schoolbook multiplication: `rows * other.columns * columns` field multiplications.
    fn mul(self, other: &Matrix<F>) -> Matrix<F> {
        assert_eq!(self.columns, other.rows, "Matrices cannot be multiplied");
        let mut result = Matrix::zeros(self.rows, other.columns);
        for i in 0..self.rows {
            for k in 0..self.columns {
                let a = self[(i, k)];
                for j in 0..other.columns {
                    result[(i, j)] += a * other[(k, j)];
                }
            }
        }
        result
    }
}

/// The other three combinations of owned and borrowed operands, through the `&a op &b` one.
macro_rules! forward_binary_op {
    ($trait:ident, $method:ident) => {
        impl<F: Scalar> $trait<Matrix<F>> for Matrix<F> {
            type Output = Matrix<F>;
            fn $method(self, other: Matrix<F>) -> Matrix<F> {
                (&self).$method(&other)
            }
        }

        impl<F: Scalar> $trait<&Matrix<F>> for Matrix<F> {
            type Output = Matrix<F>;
            fn $method(self, other: &Matrix<F>) -> Matrix<F> {
                (&self).$method(other)
            }
        }

        impl<F: Scalar> $trait<Matrix<F>> for &Matrix<F> {
            type Output = Matrix<F>;
            fn $method(self, other: Matrix<F>) -> Matrix<F> {
                self.$method(&other)
            }
        }
    };
}

forward_binary_op!(Add, add);
forward_binary_op!(Sub, sub);
forward_binary_op!(Mul, mul);

impl<F: Scalar> Neg for &Matrix<F> {
    type Output = Matrix<F>;

    fn neg(self) -> Matrix<F> {
        let data = self.data.iter().map(|&x| -x).collect();
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }
}

impl<F: Scalar> Neg for Matrix<F> {
    type Output = Matrix<F>;

    fn neg(self) -> Matrix<F> {
        -&self
    }
}

impl<F: Scalar> Matrix<F> {
    /// Creates a `rows x columns` matrix from its entries, row by row.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// // [[1, 2, 3],
    /// //  [4, 5, 6]]
    /// let a: Matrix<F> = Matrix::new(2, 3, (1..=6u64).map(F::from).collect()).unwrap();
    /// assert_eq!(a[(1, 0)], F::from(4));
    ///
    /// assert!(Matrix::<F>::new(2, 3, vec![F::from(1); 5]).is_err());
    /// ```
    /// ## Errors
    /// [`MatrixError::InvalidShape`] if `data` doesn't have `rows * columns` entries.
    pub fn new(rows: usize, columns: usize, data: Vec<F>) -> Result<Matrix<F>, MatrixError> {
        if rows.checked_mul(columns) != Some(data.len()) {
            return Err(MatrixError::InvalidShape);
        }
        Ok(Matrix {
            rows,
            columns,
            data,
        })
    }

    /// Creates a matrix from a list of rows.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// let a: Matrix<F> = Matrix::from_rows(vec![
    ///     vec![F::from(1), F::from(2)],
    ///     vec![F::from(3), F::from(4)],
    /// ])
    /// .unwrap();
    /// assert_eq!(a[(1, 0)], F::from(3));
    /// ```
    /// ## Errors
    /// [`MatrixError::InvalidShape`] if the rows don't all have the same length.
    pub fn from_rows(rows: Vec<Vec<F>>) -> Result<Matrix<F>, MatrixError> {
        let columns = rows.first().map_or(0, |row| row.len());
        if rows.iter().any(|row| row.len() != columns) {
            return Err(MatrixError::InvalidShape);
        }
        Ok(Matrix {
            rows: rows.len(),
            columns,
            data: rows.into_iter().flatten().collect(),
        })
    }

    /// The `rows x columns` matrix of zeros.
    pub fn zeros(rows: usize, columns: usize) -> Matrix<F> {
        Matrix {
            rows,
            columns,
            data: vec![F::ZERO_ELEMENT; rows * columns],
        }
    }

    /// The `n x n` identity matrix.
    pub fn identity(n: usize) -> Matrix<F> {
        Matrix::from_fn(n, n, |i, j| {
            if i == j {
                F::ONE_ELEMENT
            } else {
                F::ZERO_ELEMENT
            }
        })
    }

    /// The `rows x columns` matrix whose entry `(i, j)` is `f(i, j)`.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # use ark_ff::Field;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// // A Vandermonde matrix: row i holds the powers of x_i = i + 1.
    /// let v: Matrix<F> = Matrix::from_fn(3, 3, |i, j| F::from(i as u64 + 1).pow([j as u64]));
    /// assert_eq!(v[(2, 2)], F::from(9));
    /// ```
    pub fn from_fn(rows: usize, columns: usize, mut f: impl FnMut(usize, usize) -> F) -> Matrix<F> {
        let data = (0..rows * columns)
            .map(|k| f(k / columns, k % columns))
            .collect();
        Matrix {
            rows,
            columns,
            data,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The entries, row by row.
    pub fn as_slice(&self) -> &[F] {
        &self.data
    }

    /// The entries as a list of rows (the format [`Matrix::from_rows`] takes).
    pub fn to_rows(&self) -> Vec<Vec<F>> {
        if self.columns == 0 {
            return vec![Vec::new(); self.rows];
        }
        self.data.chunks(self.columns).map(<[F]>::to_vec).collect()
    }

    /// The entry at row `i`, column `j`, or `None` if it is outside the matrix. `a[(i, j)]` is
    /// the same but panics.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]).unwrap();
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// assert_eq!(a.get(0, 1), Some(F::from(2)));
    /// assert_eq!(a.get(2, 0), None);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> Option<F> {
        (i < self.rows && j < self.columns).then(|| self.data[i * self.columns + j])
    }

    /// Sets the entry at row `i`, column `j`. `a[(i, j)] = value` is the same but panics.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let mut a: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]).unwrap();
    /// a.set(0, 0, F::from(9)).unwrap();
    /// assert_eq!(a[(0, 0)], F::from(9));
    /// ```
    /// ## Errors
    /// [`MatrixError::OutOfBounds`] if `(i, j)` is outside the matrix.
    pub fn set(&mut self, i: usize, j: usize, value: F) -> Result<(), MatrixError> {
        if i >= self.rows || j >= self.columns {
            return Err(MatrixError::OutOfBounds);
        }
        self.data[i * self.columns + j] = value;
        Ok(())
    }

    /// Whether the matrix has as many rows as columns (the empty 0 x 0 matrix included).
    pub fn is_square(&self) -> bool {
        self.rows == self.columns
    }

    /// Whether every entry off the main diagonal is zero.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// let d: Matrix<F> = Matrix::from_fn(3, 3, |i, j| if i == j { F::from(i as u64 + 2) } else { F::from(0) });
    /// assert!(d.is_diagonal());
    /// ```
    pub fn is_diagonal(&self) -> bool {
        (0..self.rows).all(|i| (0..self.columns).all(|j| i == j || self[(i, j)] == F::ZERO_ELEMENT))
    }

    /// Whether this is an identity matrix: square, ones on the diagonal, zeros elsewhere.
    pub fn is_identity(&self) -> bool {
        self.is_square() && *self == Matrix::identity(self.rows)
    }

    /// The transpose: entry `(i, j)` becomes entry `(j, i)`.
    pub fn transpose(&self) -> Matrix<F> {
        Matrix::from_fn(self.columns, self.rows, |i, j| self[(j, i)])
    }

    /// Every entry multiplied by `scalar`.
    pub fn scalar_mul(&self, scalar: F) -> Matrix<F> {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }

    /// The matrix times a column vector: `A * v`.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]).unwrap();
    /// // a = [[1, 2], [3, 4]]: [1 * 1 + 2 * 2, 3 * 1 + 4 * 2]
    /// assert_eq!(a.mul_vec(&[F::from(1), F::from(2)]), Ok(vec![F::from(5), F::from(11)]));
    /// ```
    /// ## Errors
    /// [`MatrixError::DimensionMismatch`] if `v` doesn't have one entry per column.
    pub fn mul_vec(&self, v: &[F]) -> Result<Vec<F>, MatrixError> {
        if v.len() != self.columns {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok((0..self.rows)
            .map(|i| (0..self.columns).fold(F::ZERO_ELEMENT, |acc, j| acc + self[(i, j)] * v[j]))
            .collect())
    }

    /// The sum of all entries.
    pub fn sum(&self) -> F {
        self.data.iter().fold(F::ZERO_ELEMENT, |acc, &x| acc + x)
    }

    /// The determinant, by Gaussian elimination: about n^3 / 3 field multiplications.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]).unwrap();
    /// // a = [[1, 2], [3, 4]]: 1 * 4 - 2 * 3
    /// assert_eq!(a.determinant(), Ok(-F::from(2)));
    /// ```
    /// ## Errors
    /// [`MatrixError::NotSquare`].
    pub fn determinant(&self) -> Result<F, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        let n = self.rows;
        let mut a = self.data.clone();
        let mut det = F::ONE_ELEMENT;

        for i in 0..n {
            let Some(pivot_row) = (i..n).find(|&r| a[r * n + i] != F::ZERO_ELEMENT) else {
                return Ok(F::ZERO_ELEMENT);
            };
            if pivot_row != i {
                for k in 0..n {
                    a.swap(i * n + k, pivot_row * n + k);
                }
                det = -det;
            }

            let pivot = a[i * n + i];
            det *= pivot;
            let pivot_inverse = pivot.invert().expect("pivot is not zero");

            for r in (i + 1)..n {
                let factor = a[r * n + i] * pivot_inverse;
                if factor != F::ZERO_ELEMENT {
                    for k in (i + 1)..n {
                        let value = a[i * n + k];
                        a[r * n + k] -= factor * value;
                    }
                }
            }
        }

        Ok(det)
    }

    /// The inverse, by Gauss-Jordan elimination with row exchanges: about n^3 field
    /// multiplications.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::{Matrix, MatrixError};
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]).unwrap();
    /// let inverse = a.inverse().unwrap();
    /// assert!((&a * &inverse).is_identity());
    ///
    /// let singular: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(2), F::from(4)]]).unwrap();
    /// assert_eq!(singular.inverse(), Err(MatrixError::Singular));
    /// ```
    /// ## Errors
    /// [`MatrixError::NotSquare`], [`MatrixError::Singular`].
    pub fn inverse(&self) -> Result<Matrix<F>, MatrixError> {
        let n = self.rows;
        let data = self.gauss_jordan(Matrix::identity(n).data, n)?;
        Ok(Matrix {
            rows: n,
            columns: n,
            data,
        })
    }

    /// Solves `A x = b` for `x`, by Gaussian elimination with row exchanges.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]).unwrap();
    /// let b = vec![F::from(1), F::from(2)];
    /// let x = a.solve(&b).unwrap();
    /// assert_eq!(x, vec![F::from(0), F::from(1) / F::from(2)]);
    /// assert_eq!(a.mul_vec(&x), Ok(b));
    /// ```
    /// ## Errors
    /// [`MatrixError::NotSquare`], [`MatrixError::DimensionMismatch`] if `b` doesn't have one
    /// entry per row, [`MatrixError::Singular`].
    pub fn solve(&self, b: &[F]) -> Result<Vec<F>, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        if b.len() != self.rows {
            return Err(MatrixError::DimensionMismatch);
        }
        self.gauss_jordan(b.to_vec(), 1)
    }

    /// Gauss-Jordan elimination with row exchanges on `[self | rhs]`, where `rhs` is an
    /// `n x k` matrix stored row by row. Returns the solution columns, row by row.
    fn gauss_jordan(&self, mut rhs: Vec<F>, k: usize) -> Result<Vec<F>, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        let n = self.rows;
        let mut a = self.data.clone();

        for i in 0..n {
            let pivot_row = (i..n)
                .find(|&r| a[r * n + i] != F::ZERO_ELEMENT)
                .ok_or(MatrixError::Singular)?;
            if pivot_row != i {
                for c in 0..n {
                    a.swap(i * n + c, pivot_row * n + c);
                }
                for c in 0..k {
                    rhs.swap(i * k + c, pivot_row * k + c);
                }
            }

            let pivot_inverse = a[i * n + i].invert().expect("pivot is not zero");
            for c in 0..n {
                a[i * n + c] *= pivot_inverse;
            }
            for c in 0..k {
                rhs[i * k + c] *= pivot_inverse;
            }

            for r in (0..n).filter(|&r| r != i) {
                let factor = a[r * n + i];
                if factor != F::ZERO_ELEMENT {
                    for c in 0..n {
                        let value = a[i * n + c];
                        a[r * n + c] -= factor * value;
                    }
                    for c in 0..k {
                        let value = rhs[i * k + c];
                        rhs[r * k + c] -= factor * value;
                    }
                }
            }
        }

        Ok(rhs)
    }

    /// The adjugate (the transpose of the cofactor matrix), so that
    /// `a * a.adjugate() == det(a) * I`, also when `a` is singular.
    ///
    /// About n^3 field multiplications, without computing any cofactor matrix:
    /// - invertible: `adj(A) = det(A) * A^-1`;
    /// - rank n - 1: `adj(A)` has rank one, `c * x * y^T` with `A x = 0` and `y^T A = 0`; one
    ///   cofactor gives `c`;
    /// - lower rank: every cofactor is zero.
    ///
    /// [`naive::adjugate`](crate::naive::adjugate) computes it from the definition (n^2 cofactors,
    /// about n^5 / 3 multiplications) for comparison.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]).unwrap();
    /// assert_eq!(
    ///     a.adjugate(),
    ///     Matrix::from_rows(vec![vec![F::from(4), -F::from(2)], vec![-F::from(3), F::from(1)]])
    /// );
    /// ```
    /// ## Errors
    /// [`MatrixError::NotSquare`].
    pub fn adjugate(&self) -> Result<Matrix<F>, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        let n = self.rows;
        if n == 1 {
            return Ok(Matrix::identity(1));
        }
        match self.inverse() {
            Ok(inverse) => return Ok(inverse.scalar_mul(self.determinant()?)),
            Err(MatrixError::Singular) => {}
            Err(e) => return Err(e),
        }

        let (rank, x) = self.rank_and_kernel_vector();
        if rank + 1 < n {
            return Ok(Matrix::zeros(n, n));
        }
        // Rank n - 1: the kernels of A and of A^T are lines, spanned by x and y.
        let x = x.expect("rank n - 1 leaves a kernel");
        let y = self
            .transpose()
            .rank_and_kernel_vector()
            .1
            .expect("A^T has the same rank");
        let p = x
            .iter()
            .position(|&v| v != F::ZERO_ELEMENT)
            .expect("x is not zero");
        let q = y
            .iter()
            .position(|&v| v != F::ZERO_ELEMENT)
            .expect("y is not zero");
        // adj[p][q] is the cofactor of (q, p); it fixes the scale c of c * x * y^T.
        let minor = self.minor(q, p).determinant()?;
        let cofactor = if (p + q) % 2 == 0 { minor } else { -minor };
        let c = cofactor * (x[p] * y[q]).invert().expect("x[p] and y[q] are not zero");
        Ok(Matrix::from_fn(n, n, |i, j| c * x[i] * y[j]))
    }

    /// The matrix to the power `k`, by square-and-multiply: at most `2 * log2(k)` matrix
    /// multiplications. [`naive::pow`](crate::naive::pow) multiplies `k - 1` times.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// // [[1, 1], [1, 0]]^k holds Fibonacci numbers: F(11) = 89
    /// let fib: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(1)], vec![F::from(1), F::from(0)]]).unwrap();
    /// assert_eq!(fib.pow(10).unwrap()[(0, 0)], F::from(89));
    /// assert!(fib.pow(0).unwrap().is_identity());
    /// ```
    /// ## Errors
    /// [`MatrixError::NotSquare`].
    pub fn pow(&self, mut k: u64) -> Result<Matrix<F>, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        let mut result = Matrix::identity(self.rows);
        let mut base = self.clone();
        while k > 0 {
            if k & 1 == 1 {
                result = &result * &base;
            }
            k >>= 1;
            if k > 0 {
                base = &base * &base;
            }
        }
        Ok(result)
    }

    /// The matrix without row `i` and column `j`.
    pub(crate) fn minor(&self, i: usize, j: usize) -> Matrix<F> {
        Matrix::from_fn(self.rows - 1, self.columns - 1, |r, c| {
            self[(r + usize::from(r >= i), c + usize::from(c >= j))]
        })
    }

    /// The rank, and a non-zero `x` with `A x = 0` if there is one, by reduction to row echelon
    /// form.
    pub(crate) fn rank_and_kernel_vector(&self) -> (usize, Option<Vec<F>>) {
        let (rows, columns) = (self.rows, self.columns);
        let mut a = self.data.clone();
        let mut pivots = Vec::new();
        for c in 0..columns {
            let r = pivots.len();
            if r == rows {
                break;
            }
            let Some(p) = (r..rows).find(|&p| a[p * columns + c] != F::ZERO_ELEMENT) else {
                continue;
            };
            for k in 0..columns {
                a.swap(r * columns + k, p * columns + k);
            }
            let pivot_inverse = a[r * columns + c].invert().expect("pivot is not zero");
            for k in 0..columns {
                a[r * columns + k] *= pivot_inverse;
            }
            for other in (0..rows).filter(|&other| other != r) {
                let factor = a[other * columns + c];
                if factor != F::ZERO_ELEMENT {
                    for k in 0..columns {
                        let value = a[r * columns + k];
                        a[other * columns + k] -= factor * value;
                    }
                }
            }
            pivots.push(c);
        }

        // A column without a pivot is free: set it to 1, and solve each pivot variable from its
        // row of the reduced form.
        let kernel = (0..columns).find(|c| !pivots.contains(c)).map(|free| {
            let mut x = vec![F::ZERO_ELEMENT; columns];
            x[free] = F::ONE_ELEMENT;
            for (r, &pivot) in pivots.iter().enumerate() {
                x[pivot] = -a[r * columns + free];
            }
            x
        });
        (pivots.len(), kernel)
    }

    /// The LU decomposition `A = L * U` (Doolittle: `L` lower triangular with ones on its
    /// diagonal, `U` upper triangular), without row exchanges.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::{Matrix, MatrixError};
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::from_rows(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]).unwrap();
    /// let (l, u) = a.lu_decomposition().unwrap();
    /// assert_eq!(&l * &u, a);
    ///
    /// let swap: Matrix<F> = Matrix::from_rows(vec![vec![F::from(0), F::from(1)], vec![F::from(1), F::from(0)]]).unwrap();
    /// assert_eq!(swap.lu_decomposition(), Err(MatrixError::ZeroPivot));
    /// ```
    /// ## Errors
    /// [`MatrixError::NotSquare`]; [`MatrixError::ZeroPivot`] if a pivot is zero while rows below
    /// still need it (the matrix needs row exchanges). A zero last pivot is fine: singular
    /// matrices such as `[[1, 2], [2, 4]]` have an LU decomposition.
    pub fn lu_decomposition(&self) -> Result<(Matrix<F>, Matrix<F>), MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        let n = self.rows;
        let mut l = Matrix::identity(n);
        let mut u = Matrix::zeros(n, n);

        for i in 0..n {
            for j in i..n {
                let sum = (0..i).fold(F::ZERO_ELEMENT, |acc, k| acc + l[(i, k)] * u[(k, j)]);
                u[(i, j)] = self[(i, j)] - sum;
            }
            if i + 1 < n {
                let pivot_inverse = u[(i, i)].invert().ok_or(MatrixError::ZeroPivot)?;
                for j in (i + 1)..n {
                    let sum = (0..i).fold(F::ZERO_ELEMENT, |acc, k| acc + l[(j, k)] * u[(k, i)]);
                    l[(j, i)] = (self[(j, i)] - sum) * pivot_inverse;
                }
            }
        }

        Ok((l, u))
    }

    fn assert_same_size(&self, other: &Matrix<F>) {
        assert_eq!(
            self.rows, other.rows,
            "Matrices have different number of rows"
        );
        assert_eq!(
            self.columns, other.columns,
            "Matrices have different number of columns"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{Matrix, MatrixError};
    use ark_ff::{AdditiveGroup, Field, Fp64, MontBackend};

    #[derive(ark_ff::MontConfig)]
    #[modulus = "127"]
    #[generator = "6"]
    pub struct F127Config;
    type F = Fp64<MontBackend<F127Config, 1>>;

    fn m(rows: &[&[i64]]) -> Matrix<F> {
        Matrix::from_rows(
            rows.iter()
                .map(|row| row.iter().map(|&x| F::from(x)).collect())
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn construction() {
        assert_eq!(
            Matrix::new(2, 2, vec![F::ONE; 4]),
            Ok(m(&[&[1, 1], &[1, 1]]))
        );
        assert_eq!(
            Matrix::new(2, 2, vec![F::ONE; 3]),
            Err(MatrixError::InvalidShape)
        );
        assert_eq!(
            Matrix::from_rows(vec![vec![F::ONE, F::ONE], vec![F::ONE]]),
            Err(MatrixError::InvalidShape)
        );
        assert_eq!(Matrix::<F>::zeros(2, 3).as_slice(), &[F::ZERO; 6]);
        assert_eq!(Matrix::<F>::identity(2), m(&[&[1, 0], &[0, 1]]));
        assert_eq!(
            Matrix::from_fn(2, 2, |i, j| F::from((i * 2 + j) as u64)),
            m(&[&[0, 1], &[2, 3]])
        );

        let a = m(&[&[1, 2, 3], &[4, 5, 6]]);
        assert_eq!((a.rows(), a.columns()), (2, 3));
        assert_eq!(Matrix::from_rows(a.to_rows()), Ok(a));
    }

    #[test]
    fn empty_matrices() {
        let empty = Matrix::<F>::from_rows(vec![]).unwrap();
        assert_eq!((empty.rows(), empty.columns()), (0, 0));
        assert!(empty.is_square() && empty.is_identity());
        assert_eq!(empty.determinant(), Ok(F::ONE));
        assert_eq!(empty.inverse(), Ok(empty.clone()));
        assert_eq!(empty.transpose(), empty);

        // 2 rows with no entries, and its transpose, 0 rows of 2 entries: different matrices
        let no_columns = Matrix::<F>::from_rows(vec![vec![], vec![]]).unwrap();
        assert_eq!((no_columns.rows(), no_columns.columns()), (2, 0));
        assert_eq!(no_columns.to_rows(), vec![Vec::<F>::new(); 2]);
        let transposed = no_columns.transpose();
        assert_eq!((transposed.rows(), transposed.columns()), (0, 2));
    }

    #[test]
    fn access() {
        let mut a = m(&[&[1, 2], &[3, 4]]);
        assert_eq!(a.get(1, 0), Some(F::from(3)));
        assert_eq!(a.get(0, 2), None);
        assert_eq!(a[(0, 1)], F::from(2));
        assert_eq!(a.set(1, 1, F::from(9)), Ok(()));
        assert_eq!(a.set(2, 0, F::from(9)), Err(MatrixError::OutOfBounds));
        a[(0, 0)] = F::from(7);
        assert_eq!(a, m(&[&[7, 2], &[3, 9]]));
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn index_out_of_bounds() {
        let a = m(&[&[1, 2], &[3, 4]]);
        let _ = a[(0, 2)];
    }

    #[test]
    fn predicates() {
        assert!(m(&[&[1, 2], &[3, 4]]).is_square());
        assert!(!m(&[&[1, 2], &[3, 4], &[5, 6]]).is_square());
        assert!(m(&[&[1, 0], &[0, 4]]).is_diagonal());
        assert!(!m(&[&[1, 2], &[3, 4]]).is_diagonal());
        assert!(m(&[&[1, 0, 0], &[0, 4, 0]]).is_diagonal());
        assert!(m(&[&[1, 0], &[0, 1]]).is_identity());
        assert!(!m(&[&[1, 0], &[0, 2]]).is_identity());
        assert!(!m(&[&[1, 0, 0], &[0, 1, 0]]).is_identity());
    }

    #[test]
    fn arithmetic() {
        let a = m(&[&[1, 2], &[3, 4]]);
        let b = m(&[&[2, 3], &[4, 5]]);
        assert_eq!(&a + &b, m(&[&[3, 5], &[7, 9]]));
        assert_eq!(&a - &a, Matrix::zeros(2, 2));
        assert_eq!(&a * &b, m(&[&[10, 13], &[22, 29]]));
        assert_eq!(&b * &a, m(&[&[11, 16], &[19, 28]]));
        assert_eq!(-&a, m(&[&[-1, -2], &[-3, -4]]));
        assert_eq!(a.scalar_mul(F::from(2)), m(&[&[2, 4], &[6, 8]]));
        assert_eq!(a.sum(), F::from(10));
        assert_eq!(a.transpose(), m(&[&[1, 3], &[2, 4]]));

        // every owned / borrowed combination
        assert_eq!(a.clone() + b.clone(), &a + &b);
        assert_eq!(a.clone() + &b, &a + &b);
        assert_eq!(&a + b.clone(), &a + &b);
        assert_eq!(-a.clone(), -&a);

        // rectangular
        let r = m(&[&[1, 2, 3], &[4, 5, 6]]);
        assert_eq!(&r * &r.transpose(), m(&[&[14, 32], &[32, 77]]));
        assert_eq!(
            r.mul_vec(&[F::from(1), F::from(0), F::from(1)]),
            Ok(vec![F::from(4), F::from(10)])
        );
        assert_eq!(
            r.mul_vec(&[F::from(1)]),
            Err(MatrixError::DimensionMismatch)
        );
    }

    #[test]
    #[should_panic(expected = "different number of rows")]
    fn add_different_sizes() {
        let _ = m(&[&[1, 2]]) + m(&[&[1, 2], &[3, 4]]);
    }

    #[test]
    #[should_panic(expected = "cannot be multiplied")]
    fn multiply_different_sizes() {
        let _ = m(&[&[1, 2]]) * m(&[&[1, 2]]);
    }

    #[test]
    fn determinant() {
        assert_eq!(m(&[&[2, 2], &[3, 4]]).determinant(), Ok(F::from(2)));
        assert_eq!(
            m(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).determinant(),
            Ok(F::ZERO)
        );
        // needs a row exchange
        assert_eq!(m(&[&[0, 1], &[1, 0]]).determinant(), Ok(-F::ONE));
        assert_eq!(m(&[&[1, 2, 3]]).determinant(), Err(MatrixError::NotSquare));
    }

    #[test]
    fn inverse_and_solve() {
        let a = m(&[&[1, 2], &[3, 7]]);
        assert_eq!(a.inverse(), Ok(m(&[&[7, -2], &[-3, 1]])));
        assert_eq!(m(&[&[0, 1], &[1, 0]]).inverse(), Ok(m(&[&[0, 1], &[1, 0]])));
        assert_eq!(m(&[&[1, 2], &[2, 4]]).inverse(), Err(MatrixError::Singular));
        assert_eq!(m(&[&[1, 2]]).inverse(), Err(MatrixError::NotSquare));

        let b = m(&[&[1, 2], &[3, 4]]);
        assert_eq!(
            b.solve(&[F::from(1), F::from(2)]),
            Ok(vec![F::ZERO, F::ONE / F::from(2)])
        );
        assert_eq!(b.solve(&[F::ONE]), Err(MatrixError::DimensionMismatch));
        assert_eq!(
            m(&[&[1, 2], &[2, 4]]).solve(&[F::ONE, F::ONE]),
            Err(MatrixError::Singular)
        );
    }

    #[test]
    fn adjugate() {
        assert_eq!(
            m(&[&[2, 3], &[4, 3]]).adjugate(),
            Ok(m(&[&[3, -3], &[-4, 2]]))
        );
        assert_eq!(m(&[&[5]]).adjugate(), Ok(m(&[&[1]])));
        // singular: A * adj(A) = det(A) * I = 0
        let s = m(&[&[1, 2], &[2, 4]]);
        assert_eq!(&s * &s.adjugate().unwrap(), Matrix::zeros(2, 2));
        assert_eq!(m(&[&[1, 2]]).adjugate(), Err(MatrixError::NotSquare));
    }

    #[test]
    fn lu_decomposition() {
        let (l, u) = m(&[&[1, 2], &[3, 4]]).lu_decomposition().unwrap();
        assert_eq!(l, m(&[&[1, 0], &[3, 1]]));
        assert_eq!(u, m(&[&[1, 2], &[0, -2]]));

        // singular, but the zero pivot is the last one
        let s = m(&[&[1, 2], &[2, 4]]);
        let (l, u) = s.lu_decomposition().unwrap();
        assert_eq!(&l * &u, s);

        assert_eq!(
            m(&[&[0, 1], &[1, 0]]).lu_decomposition(),
            Err(MatrixError::ZeroPivot)
        );
        assert_eq!(
            m(&[&[1, 2]]).lu_decomposition(),
            Err(MatrixError::NotSquare)
        );
    }

    #[test]
    fn display() {
        let a = m(&[&[1, -1], &[30, 4]]);
        assert_eq!(a.to_string(), "[ 1, 126]\n[30,   4]\n");
        assert_eq!(
            MatrixError::Singular.to_string(),
            "the matrix is not invertible (determinant zero)"
        );
    }
}
