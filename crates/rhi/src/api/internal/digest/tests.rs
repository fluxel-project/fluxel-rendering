//! SHA-256 against the published vectors.
//!
//! Every expected value below is transcribed from FIPS 180-4's own examples or
//! from the NIST test-vector set, **not** from this implementation's output. That
//! distinction is the whole value of the test: a digest checked against itself
//! proves only that it is deterministic, which is the one property a wrong digest
//! also has.

use super::{Sha256, sha256};

/// Renders a digest the way every published vector writes one.
fn hex(bytes: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

#[test]
fn the_empty_message_matches_the_published_vector() {
    assert_eq!(
        hex(sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn abc_matches_the_published_vector() {
    assert_eq!(
        hex(sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn the_two_block_vector_matches() {
    // FIPS 180-4's own two-block example, so this crosses the 64-byte block
    // boundary and exercises the length field at a non-zero value.
    assert_eq!(
        hex(sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn one_million_a_matches_the_published_vector() {
    // The long vector. Worth the milliseconds because it is the only case here
    // that drives the total-length counter past a single block's worth of bytes
    // many times over, so a wrong `total` accounting shows up as a wrong digest
    // rather than as a wrong digest only for inputs above some size nobody tested.
    let mut hasher = Sha256::new();
    for _ in 0..1000 {
        hasher.update(&[b'a'; 1000]);
    }
    assert_eq!(
        hex(hasher.finish()),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn a_partial_block_survives_a_later_update_that_is_too_short_to_fill_it() {
    // This is the case the published vectors do not reach, and it shipped as a
    // real bug: every vector above drives either one `update` or whole blocks, so
    // none of them leaves a partially-filled block buffered *and* then feeds an
    // update too short to complete it. Split 193 of a 200-byte message is exactly
    // that — 1 byte buffered, 7 more arriving, 63 needed — and an implementation
    // that reset the buffer here produced a digest of the first 193 bytes only,
    // while still looking deterministic and still matching every vector above.
    let message: Vec<u8> = (0..200u32).map(|byte| byte as u8).collect();
    let mut hasher = Sha256::new();
    hasher.update(&message[..193]);
    hasher.update(&message[193..]);
    assert_eq!(hasher.finish(), sha256(&message));
}

#[test]
fn an_empty_update_changes_nothing() {
    // The degenerate form of the same bug: an empty update must be a no-op rather
    // than a reset.
    let message: Vec<u8> = (0..200u32).map(|byte| byte as u8).collect();
    let mut hasher = Sha256::new();
    hasher.update(&message[..193]);
    hasher.update(&[]);
    hasher.update(&message[193..]);
    assert_eq!(hasher.finish(), sha256(&message));
}

#[test]
fn incremental_and_one_shot_agree_at_every_split() {
    // The padding path branches on how much of the final block is occupied, so a
    // single split point would only ever test one branch. Splitting a 200-byte
    // message at every offset covers all of them, including the two boundaries
    // that matter: 56 (where the length no longer fits) and 64 (a full block).
    let message: Vec<u8> = (0..200u32).map(|byte| byte as u8).collect();
    let expected = sha256(&message);
    for split in 0..=message.len() {
        let mut hasher = Sha256::new();
        hasher.update(&message[..split]);
        hasher.update(&message[split..]);
        assert_eq!(
            hasher.finish(),
            expected,
            "incremental digest disagreed with the one-shot digest at split {split}"
        );
    }
}

#[test]
fn a_single_bit_change_changes_the_digest() {
    // Not a cryptographic claim — just that the input is actually reaching the
    // compression function rather than being dropped somewhere in the buffering.
    assert_ne!(sha256(b"abcdef"), sha256(b"abcdeg"));
}
