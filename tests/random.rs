//! Checks every operation against definitions on many random matrices, over a tiny field
//! (lots of singular matrices and zero pivots) and a larger one.

use ark_ff::{AdditiveGroup, Field, Fp64, MontBackend};
use field_matrix_utils::Matrix;

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

fn identity<F: Field>(n: usize) -> Matrix<F> {
    Matrix::new(
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { F::ONE } else { F::ZERO })
                    .collect()
            })
            .collect(),
    )
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

fn check_field<F: Field>(seed: u64, min_singular: usize) {
    let mut rng = Rng(seed);
    let (mut invertible, mut singular) = (0, 0);

    for _ in 0..400 {
        let n = 1 + rng.next(5) as usize;
        let rows = random_rows::<F>(&mut rng, n, n);
        let a = Matrix::new(rows.clone());
        let det = leibniz(&rows);

        assert_eq!(a.clone().determinant(), det, "determinant of {rows:?}");

        // A * adj(A) = det(A) * I, singular or not
        assert_eq!(
            a.clone() * a.clone().adjoint(),
            identity::<F>(n).scalar_mul(det),
            "adjoint of {rows:?}"
        );

        match a.clone().inverse() {
            Some(inverse) => {
                assert_ne!(det, F::ZERO, "inverse of singular {rows:?}");
                assert_eq!(
                    a.clone() * inverse.clone(),
                    identity(n),
                    "inverse of {rows:?}"
                );
                assert_eq!(inverse * a.clone(), identity(n), "inverse of {rows:?}");

                let b: Vec<F> = (0..n).map(|_| F::from(rng.next(1000))).collect();
                let x = a.clone().ax_b_solve_for_x(b.clone());
                assert_eq!(a.clone().mul_vec(x), b, "solve {rows:?}");
                invertible += 1;
            }
            None => {
                assert_eq!(det, F::ZERO, "no inverse for invertible {rows:?}");
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
        let a = Matrix::new(rows.clone());
        let b = Matrix::new(random_rows::<F127>(&mut rng, c, k));
        let v: Vec<F127> = (0..c).map(|_| F127::from(rng.next(1000))).collect();

        // A * v equals A times v as a one-column matrix
        let column = Matrix::new(v.iter().map(|&x| vec![x]).collect());
        let product = a.clone() * column;
        let as_matrix: Vec<F127> = (0..r).map(|i| product.get_element(i, 0)).collect();
        assert_eq!(a.clone().mul_vec(v), as_matrix);

        // (AB)^T = B^T A^T
        assert_eq!(
            (a.clone() * b.clone()).transpose(),
            b.transpose() * a.clone().transpose()
        );

        // scalar multiplication and equality on non-square matrices
        let three = F127::from(3u64);
        let tripled = Matrix::new(
            rows.iter()
                .map(|row| row.iter().map(|&x| x * three).collect())
                .collect(),
        );
        assert_eq!(a.clone().scalar_mul(three), tripled);
        assert_eq!(a.clone() + a.clone() + a.clone(), tripled);
        assert_eq!(tripled.clone() - a.clone(), a.clone() + a.clone());
        assert!(a == a.clone());
    }
}

#[test]
fn row_exchanges() {
    type F = F127;
    let swap = Matrix::new(vec![vec![F::ZERO, F::ONE], vec![F::ONE, F::ZERO]]);
    assert_eq!(swap.clone().inverse(), Some(swap.clone()));
    assert_eq!(
        swap.ax_b_solve_for_x(vec![F::from(5u64), F::from(9u64)]),
        vec![F::from(9u64), F::from(5u64)]
    );
}

#[test]
fn shapes() {
    type F = F127;
    let small = Matrix::new(vec![vec![F::ONE, F::ZERO], vec![F::ZERO, F::ONE]]);
    let mut big = identity::<F>(3);
    big.set_element(2, 2, F::from(9u64));
    assert!(small != big);
    assert!(!Matrix::<F>::new(vec![]).is_square());
    assert_eq!(Matrix::<F>::new(vec![]).transpose(), Matrix::new(vec![]));
}

#[test]
#[should_panic(expected = "Rows have different lengths")]
fn ragged_rows() {
    Matrix::<F127>::new(vec![vec![F127::ONE, F127::ONE], vec![F127::ONE]]);
}

#[test]
#[should_panic(expected = "without row exchanges")]
fn lu_needs_row_exchanges() {
    let swap = Matrix::new(vec![
        vec![F127::ZERO, F127::ONE],
        vec![F127::ONE, F127::ZERO],
    ]);
    swap.lu_decomposition();
}
