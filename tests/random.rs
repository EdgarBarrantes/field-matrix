//! Checks every operation against definitions on many random matrices, over a tiny field
//! (lots of singular matrices and zero pivots) and a larger one.

use ark_ff::{AdditiveGroup, Field, Fp64, MontBackend};
use field_matrix_utils::{Matrix, MatrixError};

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

/// Small deterministic generator, so failures are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }
}

fn random_rows<F: Field>(rng: &mut Rng, rows: usize, columns: usize) -> Vec<Vec<F>> {
    (0..rows)
        .map(|_| (0..columns).map(|_| F::from(rng.next(1000))).collect())
        .collect()
}

/// Determinant by the permutation (Leibniz) formula, independent of the library.
fn leibniz<F: Field>(rows: &[Vec<F>]) -> F {
    fn permute<F: Field>(rows: &[Vec<F>], used: &mut Vec<usize>, total: &mut F) {
        let n = rows.len();
        if used.len() == n {
            let inversions = (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .filter(|&(i, j)| used[i] > used[j])
                .count();
            let product = (0..n).fold(F::ONE, |p, i| p * rows[i][used[i]]);
            *total += if inversions % 2 == 0 {
                product
            } else {
                -product
            };
            return;
        }
        for c in 0..n {
            if !used.contains(&c) {
                used.push(c);
                permute(rows, used, total);
                used.pop();
            }
        }
    }
    let mut total = F::ZERO;
    permute(rows, &mut Vec::new(), &mut total);
    total
}

/// The top-left k x k corner.
fn leading_minor<F: Field>(rows: &[Vec<F>], k: usize) -> Vec<Vec<F>> {
    rows[..k].iter().map(|row| row[..k].to_vec()).collect()
}

fn check_field<F: Field>(seed: u64, min_singular: usize) {
    let mut rng = Rng(seed);
    let (mut invertible, mut singular) = (0, 0);

    for _ in 0..400 {
        let n = 1 + rng.next(5) as usize;
        let rows = random_rows::<F>(&mut rng, n, n);
        let a = Matrix::from_rows(rows.clone()).unwrap();
        let det = leibniz(&rows);
        let identity = Matrix::<F>::identity(n);

        assert_eq!(a.determinant(), Ok(det), "determinant of {rows:?}");

        // A * adj(A) = det(A) * I, singular or not
        assert_eq!(
            &a * &a.adjugate().unwrap(),
            identity.scalar_mul(det),
            "adjugate of {rows:?}"
        );

        // LU without row exchanges: exists exactly when no leading minor before the last is zero
        let needs_exchange = (1..n).any(|k| leibniz(&leading_minor(&rows, k)) == F::ZERO);
        match a.lu_decomposition() {
            Ok((l, u)) => {
                assert!(!needs_exchange, "LU of {rows:?}");
                assert_eq!(&l * &u, a, "LU of {rows:?}");
                for i in 0..n {
                    assert_eq!(l[(i, i)], F::ONE);
                    for j in (i + 1)..n {
                        assert_eq!(l[(i, j)], F::ZERO);
                        assert_eq!(u[(j, i)], F::ZERO);
                    }
                }
            }
            Err(e) => {
                assert_eq!(e, MatrixError::ZeroPivot);
                assert!(needs_exchange, "no LU for {rows:?}");
            }
        }

        match a.inverse() {
            Ok(inverse) => {
                assert_ne!(det, F::ZERO, "inverse of singular {rows:?}");
                assert_eq!(&a * &inverse, identity, "inverse of {rows:?}");
                assert_eq!(&inverse * &a, identity, "inverse of {rows:?}");

                let b: Vec<F> = (0..n).map(|_| F::from(rng.next(1000))).collect();
                let x = a.solve(&b).unwrap();
                assert_eq!(a.mul_vec(&x), Ok(b), "solve {rows:?}");
                invertible += 1;
            }
            Err(e) => {
                assert_eq!(e, MatrixError::Singular);
                assert_eq!(det, F::ZERO, "no inverse for invertible {rows:?}");
                assert_eq!(a.solve(&vec![F::ONE; n]), Err(MatrixError::Singular));
                singular += 1;
            }
        }
    }
    assert!(
        invertible > 50 && singular >= min_singular,
        "{invertible} invertible, {singular} singular"
    );
}

#[test]
fn random_square_matrices_f7() {
    check_field::<F7>(0x9e37_79b9_7f4a_7c15, 30);
}

#[test]
fn random_square_matrices_f127() {
    check_field::<F127>(0x2545_f491_4f6c_dd1d, 0);
}

#[test]
fn random_rectangular_matrices() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for _ in 0..200 {
        let (r, c, k) = (
            1 + rng.next(4) as usize,
            1 + rng.next(4) as usize,
            1 + rng.next(4) as usize,
        );
        let rows = random_rows::<F127>(&mut rng, r, c);
        let a = Matrix::from_rows(rows.clone()).unwrap();
        let b = Matrix::from_rows(random_rows::<F127>(&mut rng, c, k)).unwrap();
        let v: Vec<F127> = (0..c).map(|_| F127::from(rng.next(1000))).collect();

        // A * v equals A times v as a one-column matrix
        let column = Matrix::new(c, 1, v.clone()).unwrap();
        assert_eq!(a.mul_vec(&v).unwrap(), (&a * &column).as_slice());

        // (AB)^T = B^T A^T
        assert_eq!((&a * &b).transpose(), &b.transpose() * &a.transpose());

        // the flat and the row-by-row views agree
        assert_eq!(a.to_rows(), rows);
        assert_eq!(Matrix::new(r, c, a.as_slice().to_vec()), Ok(a.clone()));
        for (i, row) in rows.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                assert_eq!(a[(i, j)], x);
                assert_eq!(a.get(i, j), Some(x));
            }
        }

        // scalar multiplication and equality on non-square matrices
        let three = F127::from(3u64);
        let tripled = Matrix::from_fn(r, c, |i, j| rows[i][j] * three);
        assert_eq!(a.scalar_mul(three), tripled);
        assert_eq!(&(&a + &a) + &a, tripled);
        assert_eq!(&tripled - &a, &a + &a);
        assert_eq!(-&a + &a, Matrix::zeros(r, c));
        assert!(a == a.clone());

        // operations that need a square matrix say so
        if r != c {
            assert_eq!(a.determinant(), Err(MatrixError::NotSquare));
            assert_eq!(a.inverse(), Err(MatrixError::NotSquare));
            assert_eq!(a.adjugate(), Err(MatrixError::NotSquare));
        }
    }
}

#[test]
fn row_exchanges() {
    type F = F127;
    let swap = Matrix::from_rows(vec![vec![F::ZERO, F::ONE], vec![F::ONE, F::ZERO]]).unwrap();
    assert_eq!(swap.inverse(), Ok(swap.clone()));
    assert_eq!(
        swap.solve(&[F::from(5u64), F::from(9u64)]),
        Ok(vec![F::from(9u64), F::from(5u64)])
    );
}

#[test]
fn shapes() {
    type F = F127;
    let small = Matrix::<F>::identity(2);
    let mut big = Matrix::<F>::identity(3);
    big[(2, 2)] = F::from(9u64);
    assert!(small != big);
    // same entries, different shapes
    assert_ne!(
        Matrix::new(1, 4, vec![F::ONE; 4]).unwrap(),
        Matrix::new(2, 2, vec![F::ONE; 4]).unwrap()
    );
}
