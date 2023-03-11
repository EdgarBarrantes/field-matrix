use core::ops::{Add, Sub};
use std::fmt;

use crate::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct MatrixVector<F: Field> {
    pub vector: Vec<Matrix<F>>,
}

impl<F: Field> fmt::Display for MatrixVector<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for matrix in self.vector.iter() {
            s += &format!("{:?}\n", matrix);
        }
        write!(f, "{}", s)
    }
}

impl<F: Field> Add<MatrixVector<F>> for MatrixVector<F> {
    type Output = MatrixVector<F>;

    fn add(self, other: MatrixVector<F>) -> Self::Output {
        let mut result = MatrixVector::new_empty();
        for (m1, m2) in self.iter().zip(other.iter()) {
            result.push(m1.to_owned() + m2.to_owned());
        }

        result
    }
}

impl<F: Field> Sub<MatrixVector<F>> for MatrixVector<F> {
    type Output = MatrixVector<F>;

    fn sub(self, other: MatrixVector<F>) -> Self::Output {
        let mut result = MatrixVector::new_empty();
        for (m1, m2) in self.iter().zip(other.iter()) {
            result.push(m1.to_owned() - m2.to_owned());
        }

        result
    }
}

impl<F: Field> PartialEq for MatrixVector<F> {
    fn eq(&self, other: &Self) -> bool {
        for (m1, m2) in self.iter().zip(other.iter()) {
            if m1 != m2 {
                return false;
            }
        }

        true
    }
}

impl<F: Field> MatrixVector<F> {
    pub fn new(vector: Vec<Matrix<F>>) -> Self {
        Self { vector }
    }

    pub fn new_empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn get(&self, index: usize) -> &Matrix<F> {
        &self.vector[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut Matrix<F> {
        &mut self.vector[index]
    }

    pub fn push(&mut self, matrix: Matrix<F>) {
        self.vector.push(matrix);
    }

    pub fn pop(&mut self) -> Option<Matrix<F>> {
        self.vector.pop()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Matrix<F>> {
        self.vector.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Matrix<F>> {
        self.vector.iter_mut()
    }

    pub fn into_iter(self) -> impl Iterator<Item = Matrix<F>> {
        self.vector.into_iter()
    }

    /// Multiplies each matrix of the vector for a matrix on the right side.
    /// The result is a new vector.
    ///
    pub fn right_multiply(&self, matrix: &Matrix<F>) -> MatrixVector<F> {
        let mut result = MatrixVector::new_empty();
        for m in self.iter() {
            result.push(m.to_owned() * matrix.to_owned());
        }

        result
    }

    /// Multiplies each matrix of the vector for a matrix on the left side.
    /// The result is a new vector.
    ///
    pub fn left_multiply(&self, matrix: &Matrix<F>) -> MatrixVector<F> {
        let mut result = MatrixVector::new_empty();
        for m in self.iter() {
            result.push(matrix.to_owned() * m.to_owned());
        }

        result
    }

    /// Evaluates the matrix for alpha.
    /// Each matrix is multiplied by alpha to the power of its index.
    /// The result is a new vector.
    pub fn multiply_by_alpha(&self, alpha: F) -> MatrixVector<F> {
        let mut result = MatrixVector::new(vec![self.get(0).clone()]);
        let powers = alpha.powers();
        for (m, alpha_i) in self.iter().zip(powers).skip(1) {
            result.push(m.to_owned().scalar_mul(alpha_i));
        }
        result
    }

    /// Multiplies each matrix of the vector for a vector.
    /// The result is a new vector of vectors.
    pub fn multiply_by_vector(&self, vector: &Vec<F>) -> VectorOfVectors<F> {
        let mut result = VectorOfVectors::new_empty();
        for m in self.iter() {
            result.push(Vector::new(m.to_owned().mul_vec(vector)));
        }
        result
    }

    /// Evaluates the matrix for alpha.
    /// Where the return is the sumation of all matrices in the vector.
    pub fn evaluate(&self, alpha: F) -> Matrix<F> {
        // let result = self.get(0)
        let alpha_mul = self.multiply_by_alpha(alpha);
        let mut result = alpha_mul.get(0).to_owned();
        for m in alpha_mul.iter().skip(1) {
            result = result + m.to_owned();
        }

        result
    }

    pub fn transpose_all(&self) -> MatrixVector<F> {
        let mut result = MatrixVector::new_empty();
        for m in self.iter() {
            result.push(m.to_owned().transpose());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    use super::*;
    use ark_ff::{Fp64, MontBackend};

    #[test]
    fn test_matrix_vector_matrix_vector_right_multiply() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = Matrix::new(vec![
            Vector::new(vec![F::from(1), F::from(2), F::from(3)]).vector(),
            Vector::new(vec![F::from(4), F::from(5), F::from(6)]).vector(),
            Vector::new(vec![F::from(7), F::from(8), F::from(9)]).vector(),
        ]);
        let matrix_vector = MatrixVector::new(vec![a.clone(), a.clone()]);
        assert_eq!(
            matrix_vector.right_multiply(&a),
            MatrixVector::new(vec![
                Matrix::new(vec![
                    vec![F::from(30), F::from(36), F::from(42),],
                    vec![F::from(66), F::from(81), F::from(96),],
                    vec![F::from(102), F::from(126), F::from(150),]
                ]),
                Matrix::new(vec![
                    vec![F::from(30), F::from(36), F::from(42),],
                    vec![F::from(66), F::from(81), F::from(96),],
                    vec![F::from(102), F::from(126), F::from(150),]
                ])
            ])
        );
    }

    #[test]
    fn test_matrix_vector_matrix_vector_multiply_by_alpha() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = Matrix::new(vec![
            Vector::new(vec![F::from(1), F::from(2), F::from(3)]).vector(),
            Vector::new(vec![F::from(4), F::from(5), F::from(6)]).vector(),
            Vector::new(vec![F::from(7), F::from(8), F::from(9)]).vector(),
        ]);
        let matrix_vector = MatrixVector::new(vec![a.clone(), a.clone(), a.clone()]);
        assert_eq!(
            matrix_vector.multiply_by_alpha(F::from(2)),
            MatrixVector::new(vec![
                Matrix::new(vec![
                    vec![F::from(1), F::from(2), F::from(3),],
                    vec![F::from(4), F::from(5), F::from(6),],
                    vec![F::from(7), F::from(8), F::from(9),]
                ]),
                Matrix::new(vec![
                    vec![F::from(2), F::from(4), F::from(6),],
                    vec![F::from(8), F::from(10), F::from(12),],
                    vec![F::from(14), F::from(16), F::from(18),]
                ]),
                Matrix::new(vec![
                    vec![F::from(4), F::from(8), F::from(12),],
                    vec![F::from(16), F::from(20), F::from(24),],
                    vec![F::from(28), F::from(32), F::from(36),]
                ])
            ])
        );
    }

    #[test]
    fn test_matrix_vector_matrix_vector_multiply_by_vector() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = Matrix::new(vec![
            Vector::new(vec![F::from(1), F::from(2), F::from(3)]).vector(),
            Vector::new(vec![F::from(4), F::from(5), F::from(6)]).vector(),
            Vector::new(vec![F::from(7), F::from(8), F::from(9)]).vector(),
        ]);
        let result_vector = Vector::new(vec![F::from(5), F::from(14), F::from(23)]);
        let result_vector_of_vectors = VectorOfVectors::new(vec![
            result_vector.clone(),
            result_vector.clone(),
            result_vector.clone(),
        ]);
        let matrix_vector = MatrixVector::new(vec![a.clone(), a.clone(), a.clone()]);
        let vector_to_multiply_by = vec![F::from(1), F::from(2), F::from(0)];
        assert_eq!(
            matrix_vector.multiply_by_vector(&vector_to_multiply_by),
            result_vector_of_vectors
        );
    }

    #[test]
    fn test_matrix_vector_matrix_vector_matrix_evaulate() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = Matrix::new(vec![
            Vector::new(vec![F::from(1), F::from(2), F::from(3)]).vector(),
            Vector::new(vec![F::from(4), F::from(5), F::from(6)]).vector(),
            Vector::new(vec![F::from(7), F::from(8), F::from(9)]).vector(),
        ]);
        let matrix_vector = MatrixVector::new(vec![a.clone(), a.clone(), a.clone()]);
        let result_matrix = Matrix::new(vec![
            Vector::new(vec![F::from(7), F::from(14), F::from(21)]).vector(),
            Vector::new(vec![F::from(28), F::from(35), F::from(42)]).vector(),
            Vector::new(vec![F::from(49), F::from(56), F::from(63)]).vector(),
        ]);
        assert_eq!(matrix_vector.evaluate(F::from(2)), result_matrix);
    }

    #[test]
    fn test_matrix_vector_vector_of_vectors_sum_of_alpha_vectors() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = Vector::new(vec![F::from(1), F::from(2), F::from(3)]);
        let b = Vector::new(vec![F::from(4), F::from(5), F::from(6)]);
        let c = Vector::new(vec![F::from(7), F::from(8), F::from(9)]);
        let vector_of_vectors = VectorOfVectors::new(vec![a.clone(), b.clone(), c.clone()]);
        assert_eq!(
            vector_of_vectors.evualuate(F::from(2)),
            Vector::new(vec![F::from(37), F::from(44), F::from(51),])
        );
    }

    #[test]
    fn test_matrix_vector_vector_of_vectors_multiply_by_alpha() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = Vector::new(vec![F::from(1), F::from(2), F::from(3)]);
        let b = Vector::new(vec![F::from(4), F::from(5), F::from(6)]);
        let c = Vector::new(vec![F::from(7), F::from(8), F::from(9)]);
        let vector_of_vectors = VectorOfVectors::new(vec![a.clone(), b.clone(), c.clone()]);
        assert_eq!(
            vector_of_vectors.multiply_by_alpha(F::from(2)),
            VectorOfVectors::new(vec![
                Vector::new(vec![F::from(1), F::from(2), F::from(3),]),
                Vector::new(vec![F::from(8), F::from(10), F::from(12),]),
                Vector::new(vec![F::from(28), F::from(32), F::from(36),])
            ])
        );
    }
}
