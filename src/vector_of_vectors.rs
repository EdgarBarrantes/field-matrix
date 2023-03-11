use core::ops::{Add, Sub};
use std::fmt;

use crate::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct VectorOfVectors<F: Field> {
    pub vector: Vec<Vector<F>>,
}

impl<F: Field> fmt::Display for VectorOfVectors<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for vector in self.vector.iter() {
            s += &format!("{:?}\n", vector);
        }
        write!(f, "{}", s)
    }
}

impl<F: Field> Add<VectorOfVectors<F>> for VectorOfVectors<F> {
    type Output = VectorOfVectors<F>;

    fn add(self, other: VectorOfVectors<F>) -> Self::Output {
        let mut result = self.clone();

        for i in 0..self.len() {
            result.replace(i, self.get(i).to_owned() + other.get(i).to_owned());
        }

        result
    }
}

impl<F: Field> Sub<VectorOfVectors<F>> for VectorOfVectors<F> {
    type Output = VectorOfVectors<F>;

    fn sub(self, other: VectorOfVectors<F>) -> Self::Output {
        let mut result = self.clone();

        for i in 0..self.len() {
            result.replace(i, self.get(i).to_owned() - other.get(i).to_owned());
        }

        result
    }
}

impl<F: Field> PartialEq for VectorOfVectors<F> {
    fn eq(&self, other: &Self) -> bool {
        for (v1, v2) in self.iter().zip(other.iter()) {
            if v1 != v2 {
                return false;
            }
        }

        true
    }
}

impl<F: Field> VectorOfVectors<F> {
    pub fn new(vector: Vec<Vector<F>>) -> Self {
        Self { vector }
    }

    pub fn new_empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn get(&self, index: usize) -> &Vector<F> {
        &self.vector[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut Vector<F> {
        &mut self.vector[index]
    }

    pub fn push(&mut self, vector: Vector<F>) {
        self.vector.push(vector);
    }

    pub fn pop(&mut self) -> Option<Vector<F>> {
        self.vector.pop()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vector<F>> {
        self.vector.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vector<F>> {
        self.vector.iter_mut()
    }

    pub fn into_iter(self) -> impl Iterator<Item = Vector<F>> {
        self.vector.into_iter()
    }

    pub fn replace(&mut self, index: usize, vector: Vector<F>) {
        self.vector[index] = vector;
    }

    pub fn scalar_mul(&self, scalar: F) -> Self {
        let mut result = VectorOfVectors::new_empty();
        for v in self.iter() {
            result.push(v.to_owned().scalar_mul(scalar));
        }
        result
    }

    pub fn multiply_by_alpha(&self, alpha: F) -> VectorOfVectors<F> {
        let mut result = VectorOfVectors::new_empty();
        let powers = alpha.powers();
        for (v, alpha_i) in self.iter().zip(powers) {
            result.push(v.to_owned().scalar_mul(alpha_i));
        }
        result
    }

    pub fn sum_of_vectors(&self) -> Vector<F> {
        let mut result = self.get(0).to_owned();
        for v in self.iter().skip(1) {
            result = result + v.to_owned();
        }
        result
    }

    pub fn evualuate(&self, alpha: F) -> Vector<F> {
        let mut result = self.get(0).to_owned();
        let powers = alpha.powers();
        for (v, alpha_i) in self.iter().zip(powers).skip(1) {
            result = result + v.to_owned().scalar_mul(alpha_i);
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
    fn test_vector_of_vectors_sumation() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = VectorOfVectors::new(vec![
            Vector::new(vec![F::from(1), F::from(2), F::from(3)]),
            Vector::new(vec![F::from(4), F::from(5), F::from(6)]),
        ]);
        let b = VectorOfVectors::new(vec![
            Vector::new(vec![F::from(4), F::from(5), F::from(6)]),
            Vector::new(vec![F::from(1), F::from(2), F::from(3)]),
        ]);
        assert_eq!(
            a + b,
            VectorOfVectors::new(vec![
                Vector::new(vec![F::from(5), F::from(7), F::from(9),]),
                Vector::new(vec![F::from(5), F::from(7), F::from(9),]),
            ])
        );
    }

    #[test]
    fn test_vector_of_vectors_sum_of_vectors() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = VectorOfVectors::new(vec![
            Vector::new(vec![F::from(1), F::from(2), F::from(3)]),
            Vector::new(vec![F::from(4), F::from(5), F::from(6)]),
        ]);
        assert_eq!(
            a.sum_of_vectors(),
            Vector::new(vec![F::from(5), F::from(7), F::from(9),])
        );
    }

    #[test]
    fn test_vector_of_vectors_sumation_of_alpha_vectors() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = VectorOfVectors::new(vec![
            Vector::new(vec![F::from(1), F::from(2), F::from(3)]),
            Vector::new(vec![F::from(4), F::from(5), F::from(6)]),
            Vector::new(vec![F::from(1), F::from(1), F::from(2)]),
        ]);
        assert_eq!(
            a.evualuate(F::from(2)),
            Vector::new(vec![F::from(13), F::from(16), F::from(23),])
        );
    }
}
