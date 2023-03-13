use core::ops::{Add, Sub};
use std::fmt;

use ark_ff::Field;

#[derive(Debug, Clone)]
pub struct Vector<F: Field> {
    pub vector: Vec<F>,
}

impl<F: Field> fmt::Display for Vector<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for vector in self.vector.iter() {
            s += &format!("{:?}\n", vector);
        }
        write!(f, "{}", s)
    }
}

impl<F: Field> PartialEq for Vector<F> {
    fn eq(&self, other: &Self) -> bool {
        self.vector == other.vector
    }
}

impl<F: Field> Add<Vector<F>> for Vector<F> {
    type Output = Vector<F>;

    fn add(self, other: Vector<F>) -> Self::Output {
        let mut result = Vector::new_empty();
        for (f1, f2) in self.iter().zip(other.iter()) {
            result.push(f1.to_owned() + f2.to_owned());
        }

        result
    }
}

impl<F: Field> Sub<Vector<F>> for Vector<F> {
    type Output = Vector<F>;

    fn sub(self, other: Vector<F>) -> Self::Output {
        let mut result = Vector::new_empty();
        for (v1, v2) in self.iter().zip(other.iter()) {
            result.push(v1.to_owned() - v2.to_owned());
        }

        result
    }
}

impl<F: Field> Vector<F> {
    pub fn new(vector: Vec<F>) -> Self {
        Self { vector }
    }

    pub fn new_empty() -> Self {
        let vec: Vec<F> = Vec::new();
        Self::new(vec)
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn vector(&self) -> Vec<F> {
        self.vector.clone()
    }

    pub fn get(&self, index: usize) -> &F {
        &self.vector[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut F {
        &mut self.vector[index]
    }

    pub fn push(&mut self, matrix: F) {
        self.vector.push(matrix);
    }

    pub fn pop(&mut self) -> Option<F> {
        self.vector.pop()
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.vector.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.vector.iter_mut()
    }

    pub fn into_iter(self) -> impl Iterator<Item = F> {
        self.vector.into_iter()
    }

    pub fn replace(&mut self, index: usize, element: F) {
        self.vector[index] = element;
    }

    pub fn scalar_mul(&self, scalar: F) -> Self {
        let mut result = Vector::new_empty();
        for f in self.iter() {
            result.push(f.to_owned() * scalar);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Fp64, MontBackend};

    #[test]
    fn test_matrix_vector_vector_addition() {
        #[derive(ark_ff::MontConfig)]
        #[modulus = "127"]
        #[generator = "6"]
        pub struct F127Config;
        type F = Fp64<MontBackend<F127Config, 1>>;
        let a = Vector::new(vec![F::from(1), F::from(2), F::from(3)]);
        let b = Vector::new(vec![F::from(4), F::from(5), F::from(6)]);
        assert!(a + b == Vector::new(vec![F::from(5), F::from(7), F::from(9),]));
    }
}
