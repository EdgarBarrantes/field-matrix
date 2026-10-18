//! Row reduction, rank, kernel, all solutions, trace, the characteristic polynomial and
//! interpolation, checked by their defining properties on random matrices of every shape and rank,
//! over a tiny field (GF(7), where dividing by n can be dividing by zero) and a larger one.

use ark_ff::{AdditiveGroup, Field, Fp64, MontBackend};
use field_matrix_utils::polynomial::{evaluate_polynomial, interpolate};
use field_matrix_utils::{evaluate_matrix_polynomial, naive, Matrix, MatrixError};

#[derive(ark_ff::MontConfig)]
#[modulus = "7"]
#[generator = "3"]
pub struct F7Config;
type F7 = Fp64<MontBackend<F7Config, 1>>;

#[derive(ark_ff::MontConfig)]
#[modulus = "127"]
#[generator = "6"]
pub struct F127Config;
type F127 = Fp64<MontBackend<F127Config, 1>>;

struct Rng(u64);

impl Rng {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }

    fn element<F: Field>(&mut self) -> F {
        F::from(self.next(1000))
    }

    /// A random rows x columns matrix of rank at most `rank`.
    fn matrix_of_rank<F: Field>(&mut self, rows: usize, columns: usize, rank: usize) -> Matrix<F> {
        let left = Matrix::from_fn(rows, rank, |_, _| self.element());
        let right = Matrix::from_fn(rank, columns, |_, _| self.element());
        &left * &right
    }
}

fn check_row_reduction<F: Field>(seed: u64) {
    let mut rng = Rng(seed);
    for _ in 0..300 {
        let (rows, columns) = (rng.next(6) as usize, rng.next(6) as usize);
        let at_most = rng.next(rows.min(columns) as u64 + 1) as usize;
        let a: Matrix<F> = rng.matrix_of_rank(rows, columns, at_most);

        let (r, pivots) = a.reduced_row_echelon_form();
        let rank = pivots.len();
        assert!(rank <= at_most, "rank of {a}");
        assert_eq!(a.rank(), rank);
        assert_eq!(a.transpose().rank(), rank, "rank of the transpose of {a}");

        // reduced form: pivots strictly to the right, each the only non-zero in its column,
        // rows below the rank are zero, and reducing again changes nothing
        assert!(pivots.windows(2).all(|w| w[0] < w[1]));
        for (row, &c) in pivots.iter().enumerate() {
            for i in 0..rows {
                assert_eq!(r[(i, c)], if i == row { F::ONE } else { F::ZERO });
            }
        }
        for i in rank..rows {
            assert!((0..columns).all(|j| r[(i, j)] == F::ZERO));
        }
        assert_eq!(r.reduced_row_echelon_form(), (r.clone(), pivots));

        // kernel: columns - rank independent vectors, each with A x = 0
        let kernel = a.kernel();
        assert_eq!(kernel.len(), columns - rank);
        for x in &kernel {
            assert_eq!(a.mul_vec(x), Ok(vec![F::ZERO; rows]), "kernel of {a}");
        }
        if !kernel.is_empty() {
            assert_eq!(
                Matrix::from_rows(kernel.clone()).unwrap().rank(),
                kernel.len()
            );
        }

        // A x = b with b in the column space: a solution, plus the kernel
        let x0: Vec<F> = (0..columns).map(|_| rng.element()).collect();
        let b = a.mul_vec(&x0).unwrap();
        let solutions = a.solve_all(&b).unwrap().expect("b = A x0 has a solution");
        assert_eq!(a.mul_vec(&solutions.particular), Ok(b.clone()));
        assert_eq!(solutions.kernel, kernel);

        // unique solution when the matrix is square and invertible: the same as solve
        if rows == columns && rank == rows {
            assert_eq!(Some(solutions.particular), a.solve(&b).ok());
        }

        // a b outside the column space has no solution
        if rank < rows {
            let mut outside = vec![F::ZERO; rows];
            // a vector y with y^T A = 0 and y != 0 exists; any b with y . b != 0 is outside
            let y = &a.transpose().kernel()[0];
            let k = y.iter().position(|&v| v != F::ZERO).unwrap();
            outside[k] = F::ONE;
            assert_eq!(a.solve_all(&outside), Ok(None), "{a} x = {outside:?}");
        }
    }
}

fn check_characteristic_polynomial<F: Field>(seed: u64) {
    let mut rng = Rng(seed);
    for _ in 0..300 {
        let n = rng.next(7) as usize;
        let rank = rng.next(n as u64 + 1) as usize;
        let a: Matrix<F> = rng.matrix_of_rank(n, n, rank);
        let p = a.characteristic_polynomial().unwrap();

        assert_eq!(p.len(), n + 1);
        assert_eq!(p[n], F::ONE, "monic");
        let det = a.determinant().unwrap();
        assert_eq!(
            p[0],
            if n % 2 == 1 { -det } else { det },
            "constant term of {a}"
        );
        if n > 0 {
            assert_eq!(p[n - 1], -a.trace().unwrap(), "trace term of {a}");
        }

        // Cayley-Hamilton: p(A) = 0
        let terms: Vec<Matrix<F>> = p
            .iter()
            .map(|&c| Matrix::identity(n).scalar_mul(c))
            .collect();
        // sum of c_k * A^k: evaluate_matrix_polynomial needs matrix coefficients times alpha^k,
        // so do the powers of A by hand
        let mut power = Matrix::identity(n);
        let mut sum = Matrix::zeros(n, n);
        for term in &terms {
            sum = &sum + &(term * &power);
            power = &power * &a;
        }
        assert_eq!(sum, Matrix::zeros(n, n), "Cayley-Hamilton for {a}");

        // p(x) = det(x I - A) at a few points, against the Leibniz determinant
        for _ in 0..3 {
            let x: F = rng.element();
            let shifted = &Matrix::identity(n).scalar_mul(x) - &a;
            assert_eq!(
                evaluate_polynomial(&p, x),
                naive::determinant_leibniz(&shifted).unwrap()
            );
        }
    }
}

#[test]
fn row_reduction_f7() {
    check_row_reduction::<F7>(0x9e37_79b9_7f4a_7c15);
}

#[test]
fn row_reduction_f127() {
    check_row_reduction::<F127>(0x2545_f491_4f6c_dd1d);
}

#[test]
fn characteristic_polynomial_f7() {
    check_characteristic_polynomial::<F7>(0x1234_5678_9abc_def1);
}

#[test]
fn characteristic_polynomial_f127() {
    check_characteristic_polynomial::<F127>(0x0fed_cba9_8765_4321);
}

#[test]
fn interpolation() {
    let mut rng = Rng(7);
    for n in 0..8 {
        // distinct points 1, 2, ..., n
        let xs: Vec<F127> = (1..=n).map(F127::from).collect();
        let ys: Vec<F127> = (0..n).map(|_| rng.element()).collect();
        let p = interpolate(&xs, &ys).unwrap();
        assert_eq!(p.len(), n as usize);
        for (&x, &y) in xs.iter().zip(&ys) {
            assert_eq!(evaluate_polynomial(&p, x), y);
        }
        // the Vandermonde matrix evaluates the polynomial at every point
        assert_eq!(Matrix::vandermonde(&xs).mul_vec(&p), Ok(ys));
    }

    let two = F127::from(2u64);
    assert_eq!(
        interpolate(&[two, two], &[F127::ONE, F127::ZERO]),
        Err(MatrixError::Singular)
    );
    assert_eq!(
        interpolate(&[two], &[F127::ONE, F127::ZERO]),
        Err(MatrixError::DimensionMismatch)
    );
    assert_eq!(evaluate_polynomial::<F127>(&[], two), F127::ZERO);
}

#[test]
fn evaluate_matrix_polynomial_agrees_with_scalars() {
    // with 1 x 1 matrices, evaluating a matrix polynomial is evaluating a polynomial
    let p = [F127::from(3u64), F127::from(5u64), F127::from(7u64)];
    let as_matrices: Vec<Matrix<F127>> = p
        .iter()
        .map(|&c| Matrix::new(1, 1, vec![c]).unwrap())
        .collect();
    let x = F127::from(11u64);
    assert_eq!(
        evaluate_matrix_polynomial(&as_matrices, x).unwrap()[(0, 0)],
        evaluate_polynomial(&p, x)
    );
}

#[test]
fn shapes_and_errors() {
    let a = Matrix::<F127>::zeros(2, 3);
    assert_eq!(a.trace(), Err(MatrixError::NotSquare));
    assert_eq!(a.characteristic_polynomial(), Err(MatrixError::NotSquare));
    assert_eq!(
        a.solve_all(&[F127::ONE]),
        Err(MatrixError::DimensionMismatch)
    );
    assert_eq!(a.rank(), 0);
    assert_eq!(a.kernel().len(), 3);
    let empty = Matrix::<F127>::zeros(0, 0);
    assert_eq!(empty.characteristic_polynomial(), Ok(vec![F127::ONE]));
    assert_eq!(empty.trace(), Ok(F127::ZERO));
}
