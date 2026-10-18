//! The Hill cipher (1929): encryption is multiplying blocks of letters by a key matrix, decryption
//! multiplying by its inverse. And why it's broken: a few known blocks give the key away with one
//! more inverse.
//!
//!     cargo run --example hill_cipher
//!
//! The classic cipher works modulo 26, which is not a field (26 = 2 * 13), so a key needs a
//! determinant coprime to 26. Here the alphabet has 29 symbols, a prime, so arithmetic is in the
//! field GF(29) and any key with a non-zero determinant works.

use ark_ff::{Fp64, MontBackend};
use field_matrix_utils::{Matrix, MatrixError};

#[derive(ark_ff::MontConfig)]
#[modulus = "29"]
#[generator = "2"]
pub struct F29Config;
type F = Fp64<MontBackend<F29Config, 1>>;

const ALPHABET: &[u8; 29] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ .?";
const BLOCK: usize = 3;

fn encode(text: &str) -> Vec<F> {
    text.bytes()
        .map(|c| {
            F::from(
                ALPHABET
                    .iter()
                    .position(|&a| a == c)
                    .expect("in the alphabet") as u64,
            )
        })
        .collect()
}

fn decode(symbols: &[F]) -> String {
    symbols
        .iter()
        .map(|x| ALPHABET[x.to_string().parse::<usize>().unwrap()] as char)
        .collect()
}

/// Multiplies every block of 3 symbols by `key`.
fn apply(key: &Matrix<F>, symbols: &[F]) -> Vec<F> {
    symbols
        .chunks(BLOCK)
        .flat_map(|block| key.mul_vec(block).unwrap())
        .collect()
}

/// The message's first three blocks as the columns of a 3 x 3 matrix.
fn blocks_as_columns(symbols: &[F]) -> Matrix<F> {
    Matrix::from_fn(BLOCK, BLOCK, |i, j| symbols[j * BLOCK + i])
}

fn main() {
    let key = Matrix::from_rows(vec![
        vec![F::from(6u64), F::from(24u64), F::from(1u64)],
        vec![F::from(13u64), F::from(16u64), F::from(10u64)],
        vec![F::from(20u64), F::from(17u64), F::from(15u64)],
    ])
    .unwrap();
    println!("key, determinant {}:\n{key}", key.determinant().unwrap());

    let message = "ATTACK AT DAWN. ";
    let plain = encode(&message[..message.len() / BLOCK * BLOCK]);
    let cipher = apply(&key, &plain);
    println!("{:?} -> {:?}", decode(&plain), decode(&cipher));

    let decrypted = apply(&key.inverse().expect("invertible key"), &cipher);
    println!("decrypted with the inverse key: {:?}", decode(&decrypted));
    assert_eq!(decrypted, plain);

    // A key with determinant zero can't be undone: different messages encrypt the same.
    let bad_key = Matrix::from_fn(BLOCK, BLOCK, |i, j| F::from((i + j) as u64));
    assert_eq!(bad_key.inverse(), Err(MatrixError::Singular));
    println!(
        "a key with determinant 0 has no inverse: {:?}",
        bad_key.inverse().unwrap_err()
    );

    // The attack: an eavesdropper who knows 3 blocks of plaintext (P, as columns) and sees their
    // ciphertext (C = K P) recovers the key as C P^-1, if those blocks form an invertible P.
    let p = blocks_as_columns(&plain);
    let c = blocks_as_columns(&cipher);
    let recovered = &c * &p.inverse().expect("these 3 blocks are independent");
    println!("key recovered from 3 known blocks:\n{recovered}");
    assert_eq!(recovered, key);
}
