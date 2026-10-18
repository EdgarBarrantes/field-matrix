//! Every function in `naive` against its fast counterpart, on random matrices that include many
//! singular ones (rank n - 1 and lower), over a tiny field and a larger one.

use ark_ff::{Field, Fp64, MontBackend};
use field_matrix_utils::counting::{self, Counts};
use field_matrix_utils::{naive, Matrix, MatrixError};

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

    /// A random n x n matrix of rank at most `rank`: the product of n x rank and rank x n.
    fn matrix_of_rank<F: Field>(&mut self, n: usize, rank: usize) -> Matrix<F> {
        let left = Matrix::from_fn(n, rank, |_, _| F::from(self.next(1000)));
        let right = Matrix::from_fn(rank, n, |_, _| F::from(self.next(1000)));
        &left * &right
    }
}

fn check_field<F: Field>(seed: u64) -> [usize; 3] {
    let mut rng = Rng(seed);
    // how many full-rank, rank n - 1 and lower-rank matrices were checked
    let mut seen = [0; 3];

    for _ in 0..300 {
        let n = 1 + rng.next(6) as usize;
        let rank = n - (rng.next(4) as usize).min(n);
        let a: Matrix<F> = rng.matrix_of_rank(n, rank);
        let det = a.determinant().unwrap();

        assert_eq!(naive::determinant_leibniz(&a), Ok(det), "Leibniz {a}");
        assert_eq!(naive::determinant_laplace(&a), Ok(det), "Laplace {a}");
        assert_eq!(naive::adjugate(&a), a.adjugate(), "adjugate {a}");
        assert_eq!(naive::inverse(&a), a.inverse(), "inverse {a}");

        let b: Vec<F> = (0..n).map(|_| F::from(rng.next(1000))).collect();
        assert_eq!(naive::solve_cramer(&a, &b), a.solve(&b), "solve {a}");

        let k = rng.next(12);
        assert_eq!(naive::pow(&a, k), a.pow(k), "power {k} of {a}");

        let actual_rank = n - usize::from(det == F::ZERO) * {
            // the rank is below n; tell n - 1 from lower by the adjugate (zero below n - 1)
            if a.adjugate().unwrap() == Matrix::zeros(n, n) {
                2
            } else {
                1
            }
        };
        seen[(n - actual_rank).min(2)] += 1;
    }
    seen
}

#[test]
fn naive_matches_fast_f7() {
    let [full, minus_one, lower] = check_field::<F7>(0x9e37_79b9_7f4a_7c15);
    assert!(
        full > 20 && minus_one > 20 && lower > 20,
        "{full} {minus_one} {lower}"
    );
}

#[test]
fn naive_matches_fast_f127() {
    let [full, minus_one, lower] = check_field::<F127>(0x2545_f491_4f6c_dd1d);
    assert!(
        full > 20 && minus_one > 20 && lower > 20,
        "{full} {minus_one} {lower}"
    );
}

#[test]
fn not_square() {
    let a = Matrix::<F127>::zeros(2, 3);
    assert_eq!(naive::determinant_leibniz(&a), Err(MatrixError::NotSquare));
    assert_eq!(naive::determinant_laplace(&a), Err(MatrixError::NotSquare));
    assert_eq!(naive::adjugate(&a), Err(MatrixError::NotSquare));
    assert_eq!(naive::inverse(&a), Err(MatrixError::NotSquare));
    assert_eq!(
        naive::solve_cramer(&a, &[F127::ONE; 2]),
        Err(MatrixError::NotSquare)
    );
    assert_eq!(naive::pow(&a, 2), Err(MatrixError::NotSquare));
}

/// Operation counts on fixed inputs. A change that makes an operation do more (or less) work
/// fails here; update the numbers when that is the intent.
#[test]
fn operation_counts() {
    let mut rng = Rng(42);
    let a = counting::wrap(&Matrix::<F127>::from_fn(6, 6, |_, _| {
        F127::from(1 + rng.next(126))
    }));
    let counts = |additions, multiplications, inversions| Counts {
        additions,
        multiplications,
        inversions,
    };

    assert_eq!(counting::count(|| a.determinant()).1, counts(55, 76, 6));
    assert_eq!(
        counting::count(|| naive::determinant_leibniz(&a)).1,
        counts(719, 4320, 0)
    );
    assert_eq!(counting::count(|| a.adjugate()).1, counts(415, 544, 12));
    assert_eq!(
        counting::count(|| naive::adjugate(&a)).1,
        counts(1079, 1619, 180)
    );
    assert_eq!(counting::count(|| a.pow(20)).1, counts(1296, 1296, 0));
    assert_eq!(
        counting::count(|| naive::pow(&a, 20)).1,
        counts(4320, 4320, 0)
    );
}
