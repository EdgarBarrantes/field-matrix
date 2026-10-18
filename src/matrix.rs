//! Dense matrices over a finite field: [`Matrix`].

use core::ops::{Add, Mul, Sub};
use std::fmt;

use ark_ff::Field;

#[derive(Debug, Clone)]
pub struct Matrix<F: Field> {
    matrix: Vec<Vec<F>>,
}

impl<F: Field> fmt::Display for Matrix<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for row in self.matrix.iter() {
            for entry in row {
                s += &format!("{:20?}", entry);
            }
            s += "\n";
        }
        write!(f, "{}", s)
    }
}

impl<F: Field> PartialEq for Matrix<F> {
    fn eq(&self, other: &Matrix<F>) -> bool {
        self.matrix == other.matrix
    }
}

impl<F: Field> Add<Matrix<F>> for Matrix<F> {
    type Output = Matrix<F>;

    fn add(self, other: Matrix<F>) -> Matrix<F> {
        assert_eq!(
            self.matrix.len(),
            other.matrix.len(),
            "Matrices have different number of rows"
        );
        assert_eq!(
            self.num_columns(),
            other.num_columns(),
            "Matrices have different number of columns"
        );

        let mut result = self.clone();

        for i in 0..self.matrix.len() {
            for j in 0..self.num_columns() {
                result.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j];
            }
        }

        result
    }
}

impl<F: Field> Sub<Matrix<F>> for Matrix<F> {
    type Output = Matrix<F>;

    fn sub(self, other: Matrix<F>) -> Matrix<F> {
        assert_eq!(
            self.matrix.len(),
            other.matrix.len(),
            "Matrices have different number of rows"
        );
        assert_eq!(
            self.num_columns(),
            other.num_columns(),
            "Matrices have different number of columns"
        );

        let mut result = self.clone();

        for i in 0..self.matrix.len() {
            for j in 0..self.num_columns() {
                result.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j];
            }
        }

        result
    }
}

impl<F: Field> Mul<Matrix<F>> for Matrix<F> {
    type Output = Matrix<F>;

    fn mul(self, other: Matrix<F>) -> Matrix<F> {
        assert_eq!(
            self.num_columns(),
            other.matrix.len(),
            "Matrices cannot be multiplied"
        );

        let mut result = Matrix::new(vec![vec![F::ZERO; other.num_columns()]; self.matrix.len()]);

        for i in 0..self.matrix.len() {
            for j in 0..other.num_columns() {
                for k in 0..self.num_columns() {
                    result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j];
                }
            }
        }

        result
    }
}

impl<F: Field> Matrix<F> {
    /// Creates a new matrix from a vector of vectors.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// let a: Matrix<F> = Matrix::new(vec![
    ///     vec![F::from(1), F::from(2)],
    ///     vec![F::from(3), F::from(4)],
    /// ]);
    /// assert_eq!(a.get_element(1, 0), F::from(3));
    /// ```
    /// ## Panics
    /// Panics if the rows don't all have the same length.
    pub fn new(matrix: Vec<Vec<F>>) -> Matrix<F> {
        if let Some(first) = matrix.first() {
            assert!(
                matrix.iter().all(|row| row.len() == first.len()),
                "Rows have different lengths"
            );
        }
        Matrix { matrix }
    }

    fn num_columns(&self) -> usize {
        self.matrix.first().map_or(0, |row| row.len())
    }

    fn identity(n: usize) -> Matrix<F> {
        let mut matrix = vec![vec![F::ZERO; n]; n];
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = F::ONE;
        }
        Matrix { matrix }
    }

    /// Gauss-Jordan elimination with row exchanges on [self | rhs].
    /// Returns the solution columns, or None if self is singular.
    fn gauss_jordan(&self, rhs: Vec<Vec<F>>) -> Option<Vec<Vec<F>>> {
        let n = self.matrix.len();
        assert_eq!(n, self.num_columns(), "Matrix is not square");
        let mut a = self.matrix.clone();
        let mut b = rhs;

        for i in 0..n {
            let pivot_row = (i..n).find(|&r| a[r][i] != F::ZERO)?;
            a.swap(i, pivot_row);
            b.swap(i, pivot_row);

            let pivot_inverse = a[i][i].inverse()?;
            for k in 0..n {
                a[i][k] *= pivot_inverse;
            }
            for value in b[i].iter_mut() {
                *value *= pivot_inverse;
            }

            for r in 0..n {
                if r != i && a[r][i] != F::ZERO {
                    let factor = a[r][i];
                    for k in 0..n {
                        let value = a[i][k];
                        a[r][k] -= factor * value;
                    }
                    for k in 0..b[r].len() {
                        let value = b[i][k];
                        b[r][k] -= factor * value;
                    }
                }
            }
        }

        Some(b)
    }

    /// Returns whether or not the matrix is square.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// assert!(a.is_square());
    /// ```
    pub fn is_square(self) -> bool {
        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        if num_rows == 0 {
            return false;
        }

        num_rows == num_columns
    }

    /// Returns the determinant of the matrix.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// // 1 * 4 - 2 * 3
    /// assert_eq!(a.determinant(), -F::from(2));
    /// ```
    pub fn determinant(mut self) -> F {
        assert_eq!(
            self.matrix.len(),
            self.matrix[0].len(),
            "Matrix is not square"
        );

        let n = self.matrix.len();
        let mut det = F::ONE;

        for i in 0..n {
            let mut pivot_row = i;
            for j in (i + 1)..n {
                if self.matrix[j][i] != F::ZERO {
                    pivot_row = j;
                    break;
                }
            }

            if pivot_row != i {
                self.matrix.swap(i, pivot_row);
                det = -det;
            }

            let pivot = self.matrix[i][i];

            if pivot == F::ZERO {
                return F::ZERO;
            }

            det *= pivot;

            for j in (i + 1)..n {
                let factor = self.matrix[j][i] / pivot;
                for k in (i + 1)..n {
                    self.matrix[j][k] = self.matrix[j][k] - factor * self.matrix[i][k];
                }
            }
        }

        det
    }

    /// Returns whether or not the matrix is diagonal.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// assert!(!a.is_diagonal());
    /// let d: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(0)], vec![F::from(0), F::from(4)]]);
    /// assert!(d.is_diagonal());
    /// ```
    pub fn is_diagonal(self) -> bool {
        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        if num_rows == 0 {
            return false;
        }

        for i in 0..num_rows {
            for j in 0..num_columns {
                if i != j && self.matrix[i][j] != F::ZERO {
                    return false;
                }
            }
        }

        true
    }

    /// Returns the transpose of the matrix.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// let t = a.transpose();
    /// assert_eq!(t.get_element(0, 1), F::from(3));
    /// ```
    pub fn transpose(self) -> Matrix<F> {
        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        let mut new_rows = vec![vec![F::ZERO; num_rows]; num_columns];

        for i in 0..num_rows {
            for j in 0..num_columns {
                new_rows[j][i] = self.matrix[i][j];
            }
        }

        Matrix { matrix: new_rows }
    }

    /// Returns the adjoint (adjugate) of the matrix: the transpose of its cofactor matrix,
    /// so that `a * a.adjoint() == det(a) * I`, also when `a` is singular.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// assert_eq!(
    ///     a.adjoint(),
    ///     Matrix::new(vec![vec![F::from(4), -F::from(2)], vec![-F::from(3), F::from(1)]])
    /// );
    /// ```
    /// ## Panics
    /// Panics if the matrix is not square.
    /// ## Notes
    /// Versions up to 0.1.1 returned the transpose here.
    pub fn adjoint(self) -> Matrix<F> {
        let n = self.matrix.len();
        assert_eq!(n, self.num_columns(), "Matrix is not square");
        if n == 1 {
            return Matrix::new(vec![vec![F::ONE]]);
        }

        let mut adjugate = vec![vec![F::ZERO; n]; n];

        for i in 0..n {
            for j in 0..n {
                let minor: Vec<Vec<F>> = self
                    .matrix
                    .iter()
                    .enumerate()
                    .filter(|&(r, _)| r != i)
                    .map(|(_, row)| {
                        row.iter()
                            .enumerate()
                            .filter(|&(c, _)| c != j)
                            .map(|(_, &value)| value)
                            .collect()
                    })
                    .collect();
                let cofactor = Matrix { matrix: minor }.determinant();
                adjugate[j][i] = if (i + j) % 2 == 0 {
                    cofactor
                } else {
                    -cofactor
                };
            }
        }

        Matrix { matrix: adjugate }
    }

    /// Returns the inverse of the matrix, or None if it is not invertible.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// let inverse = a.clone().inverse().unwrap();
    /// assert!((a * inverse).is_identity());
    ///
    /// let singular: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(2), F::from(4)]]);
    /// assert_eq!(singular.inverse(), None);
    /// ```
    /// ## Panics
    /// Panics if the matrix is not square.
    /// ## Notes
    /// This function uses Gauss-Jordan elimination with row exchanges.
    /// Versions up to 0.1.1 panicked instead of returning None, and also on invertible
    /// matrices that need row exchanges, such as [[0, 1], [1, 0]].
    pub fn inverse(self) -> Option<Matrix<F>> {
        let n = self.matrix.len();
        self.gauss_jordan(Matrix::identity(n).matrix)
            .map(|matrix| Matrix { matrix })
    }

    /// Returns whether or not the matrix is the identity matrix.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// assert!(!a.is_identity());
    /// let i: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(0)], vec![F::from(0), F::from(1)]]);
    /// assert!(i.is_identity());
    /// ```
    /// ## Notes
    /// This function returns false if the matrix is empty.
    pub fn is_identity(self) -> bool {
        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        if num_rows == 0 {
            return false;
        }

        for i in 0..num_rows {
            for j in 0..num_columns {
                if i == j && self.matrix[i][j] != F::ONE {
                    return false;
                }
                if i != j && self.matrix[i][j] != F::ZERO {
                    return false;
                }
            }
        }

        true
    }

    /// Solves the system of linear equations Ax = b for x.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// let b = vec![F::from(1), F::from(2)];
    /// let x = a.clone().ax_b_solve_for_x(b.clone());
    /// assert_eq!(x, vec![F::from(0), F::from(1) / F::from(2)]);
    /// assert_eq!(a.mul_vec(x), b);
    /// ```
    /// ## Panics
    /// Panics if the matrix is not square, if b has the wrong length, or if the determinant
    /// of the matrix is zero.
    /// ## Notes
    /// This function uses Gaussian elimination with row exchanges.
    /// Versions up to 0.1.1 returned wrong solutions (a bug in back substitution) and
    /// panicked on matrices that need row exchanges.
    pub fn ax_b_solve_for_x(self, b: Vec<F>) -> Vec<F> {
        assert_eq!(
            b.len(),
            self.matrix.len(),
            "b must have one element per row"
        );
        let columns = b.into_iter().map(|value| vec![value]).collect();
        self.gauss_jordan(columns)
            .expect("Det(A) = 0")
            .into_iter()
            .map(|row| row[0])
            .collect()
    }

    /// Returns the LU decomposition of the matrix.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// let (l, u) = a.lu_decomposition();
    /// assert_eq!(l * u, a);
    /// ```
    /// ## Panics
    /// Panics if the matrix is not square, if its determinant is zero, or if it needs row
    /// exchanges (a zero pivot, as in [[0, 1], [1, 0]]): this decomposition has no
    /// permutation matrix.
    /// ## Notes
    /// This function uses the Doolittle algorithm to compute the LU decomposition.
    /// The Doolittle algorithm is a variant of the Gaussian elimination algorithm.
    pub fn lu_decomposition(&self) -> (Matrix<F>, Matrix<F>) {
        assert_ne!(self.clone().determinant(), F::ZERO, "Det(A) = 0");
        assert!(self.clone().is_square(), "Matrix is not square");

        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        let mut l = vec![vec![F::ZERO; num_rows]; num_columns];
        let mut u = vec![vec![F::ZERO; num_rows]; num_columns];

        for i in 0..num_rows {
            for j in 0..num_columns {
                if i == j {
                    l[i][j] = F::ONE;
                }
            }
        }

        for i in 0..num_rows {
            for j in 0..num_columns {
                let mut sum = F::ZERO;
                for k in 0..i {
                    sum += l[i][k] * u[k][j];
                }
                u[i][j] = self.matrix[i][j] - sum;
            }

            assert_ne!(
                u[i][i],
                F::ZERO,
                "Matrix has no LU decomposition without row exchanges"
            );

            for j in 0..num_columns {
                let mut sum = F::ZERO;
                for k in 0..i {
                    sum += l[j][k] * u[k][i];
                }
                l[j][i] = (self.matrix[j][i] - sum) / u[i][i];
            }
        }

        (Matrix { matrix: l }, Matrix { matrix: u })
    }

    /// Multiplies the matrix by a scalar.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// assert_eq!(a.clone().scalar_mul(F::from(2)), a.clone() + a);
    /// ```
    /// ## Notes
    /// This function is equivalent to multiplying each element of the matrix by the scalar.
    pub fn scalar_mul(self, scalar: F) -> Matrix<F> {
        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        let mut new_rows = vec![vec![F::ZERO; num_columns]; num_rows];

        for i in 0..num_rows {
            for j in 0..num_columns {
                new_rows[i][j] = self.matrix[i][j] * scalar;
            }
        }

        Matrix { matrix: new_rows }
    }

    /// Multiplies the matrix by a vector.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// // [1 * 1 + 2 * 2, 3 * 1 + 4 * 2]
    /// assert_eq!(a.mul_vec(vec![F::from(1), F::from(2)]), vec![F::from(5), F::from(11)]);
    /// ```
    /// ## Panics
    /// Panics if the number of columns in the matrix is not equal to the number of elements in
    /// the vector.
    /// ## Notes
    /// This function is equivalent to multiplying the matrix by a column vector: A * v.
    /// Versions up to 0.1.1 computed v * A (the vector as a row) instead.
    pub fn mul_vec(self, vec: Vec<F>) -> Vec<F> {
        assert_eq!(
            vec.len(),
            self.num_columns(),
            "Vector and matrix can't be multiplied by the other"
        );

        let mut result = vec![F::ZERO; self.matrix.len()];

        for i in 0..self.matrix.len() {
            for j in 0..self.num_columns() {
                result[i] += self.matrix[i][j] * vec[j];
            }
        }

        result
    }

    /// Returns the sum of all the elements in the matrix.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// assert_eq!(a.sum_of_matrix(), F::from(10));
    /// ```
    /// ## Notes
    /// This function is equivalent to summing all the elements in the matrix.
    pub fn sum_of_matrix(self) -> F {
        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        let mut sum = F::ZERO;

        for i in 0..num_rows {
            for j in 0..num_columns {
                sum += self.matrix[i][j];
            }
        }

        sum
    }

    /// Sets a specific element in the matrix.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// let mut a = a;
    /// a.set_element(0, 0, F::from(9));
    /// assert_eq!(a.get_element(0, 0), F::from(9));
    /// ```
    /// ## Panics
    /// Panics if the row or column is out of bounds.
    /// ## Notes
    /// This function is equivalent to setting the element in the matrix.
    /// With row being the position in the outer vector and column being the position in the inner vector.
    /// The first element in the outer vector is row 0 and the first element in the inner vector is column 0.
    pub fn set_element(&mut self, row: usize, column: usize, new_value: F) {
        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        assert!(
            row < num_rows && column < num_columns,
            "Index out of bounds"
        );

        self.matrix[row][column] = new_value;
    }

    /// Gets a specific element in the matrix.
    /// ## Example
    /// ```
    /// # use ark_ff::{Fp64, MontBackend};
    /// # use field_matrix_utils::Matrix;
    /// # #[derive(ark_ff::MontConfig)]
    /// # #[modulus = "127"]
    /// # #[generator = "6"]
    /// # pub struct F127Config;
    /// # type F = Fp64<MontBackend<F127Config, 1>>;
    /// # let a: Matrix<F> = Matrix::new(vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]]);
    /// // a = [[1, 2], [3, 4]] over the integers modulo 127
    /// assert_eq!(a.get_element(0, 1), F::from(2));
    /// ```
    /// ## Panics
    /// Panics if the row or column is out of bounds.
    /// ## Notes
    /// This function is equivalent to getting the element in the matrix.
    /// With row being the position in the outer vector and column being the position in the inner vector.
    /// The first element in the outer vector is row 0 and the first element in the inner vector is column 0.
    pub fn get_element(&self, row: usize, column: usize) -> F {
        let num_rows = self.matrix.len();
        let num_columns = self.num_columns();

        assert!(
            row < num_rows && column < num_columns,
            "Index out of bounds"
        );

        self.matrix[row][column]
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    use ark_ff::{Fp64, MontBackend};

    #[test]
    fn test_matrix_utils_determinant() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(2), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);
        let b: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2), F::from(3)],
            vec![F::from(4), F::from(5), F::from(6)],
            vec![F::from(7), F::from(8), F::from(9)],
        ]);

        assert_eq!(a.determinant(), F::from(2));
        assert_eq!(b.determinant(), F::from(0));
    }

    #[test]
    fn test_matrix_utils_is_square() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);
        let b = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
            vec![F::from(5), F::from(6)],
        ]);

        assert!(a.is_square());
        assert!(!b.is_square());
    }

    #[test]
    fn test_matrix_utils_is_diagonal() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(0)],
            vec![F::from(0), F::from(4)],
        ]);
        let b = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        assert!(a.is_diagonal());
        assert!(!b.is_diagonal());
    }

    #[test]
    fn test_matrix_utils_transpose() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        let b = a.transpose();

        assert_eq!(
            b,
            Matrix::new(vec![
                vec![F::from(1), F::from(3)],
                vec![F::from(2), F::from(4)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_adjoint() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(2), F::from(3)],
            vec![F::from(4), F::from(3)],
        ]);

        let b = a.adjoint();

        assert_eq!(
            b,
            Matrix::new(vec![
                vec![F::from(3), -F::from(3)],
                vec![-F::from(4), F::from(2)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_inverse() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(7)],
        ]);

        let b = a.inverse().unwrap();

        assert_eq!(
            b,
            Matrix::new(vec![
                vec![F::from(7), -F::from(2)],
                vec![-F::from(3), F::from(1)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_is_identity() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(0)],
            vec![F::from(0), F::from(1)],
        ]);
        let b = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        assert!(a.is_identity());
        assert!(!b.is_identity());
    }

    #[test]
    fn test_matrix_utils_ax_b_solve_for_x() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);
        let b = vec![F::from(1), F::from(2)];

        let x = a.ax_b_solve_for_x(b);

        // x = 0, y = 1/2: 0 + 2 * 1/2 = 1 and 0 + 4 * 1/2 = 2
        assert_eq!(x, vec![F::from(0), F::from(1) / F::from(2)]);
    }

    #[test]
    fn test_matrix_utils_lu_decomposition() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        let (l, u) = a.lu_decomposition();

        assert_eq!(
            l,
            Matrix::new(vec![
                vec![F::from(1), F::from(0)],
                vec![F::from(3), F::from(1)]
            ])
        );
        assert_eq!(
            u,
            Matrix::new(vec![
                vec![F::from(1), F::from(2)],
                vec![F::from(0), -F::from(2)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_matrix_addition() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);
        let b: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        let c = a + b;

        assert_eq!(
            c,
            Matrix::new(vec![
                vec![F::from(2), F::from(4)],
                vec![F::from(6), F::from(8)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_matrix_substraction() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);
        let b: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        let c = a - b;

        assert_eq!(
            c,
            Matrix::new(vec![
                vec![F::from(0), F::from(0)],
                vec![F::from(0), F::from(0)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_matrix_multiplication() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);
        let b: Matrix<F> = Matrix::new(vec![
            vec![F::from(2), F::from(3)],
            vec![F::from(4), F::from(5)],
        ]);

        let c = a.clone() * (b.clone());
        let d = b * (a.clone());

        assert_eq!(
            c,
            Matrix::new(vec![
                vec![F::from(10), F::from(13)],
                vec![F::from(22), F::from(29)]
            ])
        );

        assert_eq!(
            d,
            Matrix::new(vec![
                vec![F::from(11), F::from(16)],
                vec![F::from(19), F::from(28)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_matrix_scalar_multiplication() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        let b = a.scalar_mul(F::from(2));

        assert_eq!(
            b,
            Matrix::new(vec![
                vec![F::from(2), F::from(4)],
                vec![F::from(6), F::from(8)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_multiply_vec() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);
        let b = vec![F::from(1), F::from(2)];

        let c = a.mul_vec(b);

        // [1 * 1 + 2 * 2, 3 * 1 + 4 * 2]
        assert_eq!(c, vec![F::from(5), F::from(11)]);
    }

    #[test]
    fn test_matrix_utils_sum_of_matrix() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        let b = a.sum_of_matrix();

        assert_eq!(b, F::from(10));
    }

    #[test]
    fn test_matrix_utils_set_element() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let mut a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        a.set_element(0, 0, F::from(5));
        a.set_element(1, 1, F::from(6));

        assert_eq!(
            a,
            Matrix::new(vec![
                vec![F::from(5), F::from(2)],
                vec![F::from(3), F::from(6)]
            ])
        );
    }

    #[test]
    fn test_matrix_utils_get_element() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a: Matrix<F> = Matrix::new(vec![
            vec![F::from(1), F::from(2)],
            vec![F::from(3), F::from(4)],
        ]);

        assert_eq!(a.get_element(0, 0), F::from(1));
        assert_eq!(a.get_element(0, 1), F::from(2));
        assert_eq!(a.get_element(1, 0), F::from(3));
        assert_eq!(a.get_element(1, 1), F::from(4));
    }
}
