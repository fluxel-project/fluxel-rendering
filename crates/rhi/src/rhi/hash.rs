//! Deterministic canonical hashing.
//!
//! # Why this exists
//!
//! rhi-design requires two different kinds of digest and forbids confusing them:
//!
//! ```text
//! fingerprint  -> cache key candidate, logs, artifact provenance
//! compatibility -> exact interned identity, the only form correctness may use
//! ```
//!
//! Both need the *same* canonical byte encoding, because a fingerprint is only
//! useful if two semantically identical values encode identically. That encoding
//! is defined here once: every field is preceded by its type tag, and every byte
//! string is length-prefixed, so no two different structures can encode to the
//! same bytes by concatenation.
//!
//! # What it deliberately does not do
//!
//! A fingerprint never carries correctness. Callers that need exact identity
//! intern the canonical value and compare the values themselves, so a hash
//! collision cannot make two different capability contracts look equal.
//!
//! SHA-256 is implemented here rather than pulled in as a dependency: this crate
//! has no third-party runtime dependency for hashing, and the algorithm is
//! fixed, short, and testable against its published vectors.

/// A streaming canonical encoder that produces a 256-bit digest.
#[derive(Clone)]
pub(crate) struct CanonicalHasher {
    state: Sha256,
}

impl CanonicalHasher {
    /// A fresh encoder.
    pub(crate) fn new() -> Self {
        Self {
            state: Sha256::new(),
        }
    }

    /// Writes a discriminant, so two variants that share a payload encoding
    /// still encode differently.
    pub(crate) fn tag(&mut self, tag: u8) -> &mut Self {
        self.state.update(&[tag]);
        self
    }

    /// Writes one byte.
    pub(crate) fn u8(&mut self, value: u8) -> &mut Self {
        self.state.update(&[value]);
        self
    }

    /// Writes a boolean as one canonical byte.
    pub(crate) fn bool(&mut self, value: bool) -> &mut Self {
        self.u8(u8::from(value))
    }

    /// Writes a 16-bit value, little-endian.
    pub(crate) fn u16(&mut self, value: u16) -> &mut Self {
        self.state.update(&value.to_le_bytes());
        self
    }

    /// Writes a 32-bit value, little-endian.
    pub(crate) fn u32(&mut self, value: u32) -> &mut Self {
        self.state.update(&value.to_le_bytes());
        self
    }

    /// Writes a 64-bit value, little-endian.
    pub(crate) fn u64(&mut self, value: u64) -> &mut Self {
        self.state.update(&value.to_le_bytes());
        self
    }

    /// Writes a length-prefixed byte string.
    pub(crate) fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.u64(value.len() as u64);
        self.state.update(value);
        self
    }

    /// Writes a length-prefixed UTF-8 string.
    pub(crate) fn str(&mut self, value: &str) -> &mut Self {
        self.bytes(value.as_bytes())
    }

    /// Finishes the digest.
    pub(crate) fn finish(self) -> [u8; 32] {
        self.state.finish()
    }
}

/// SHA-256, per FIPS 180-4.
#[derive(Clone)]
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length_bits: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0u8; 64],
            buffered: 0,
            length_bits: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.length_bits = self.length_bits.wrapping_add((input.len() as u64) * 8);
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(input.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&input[..take]);
            self.buffered += take;
            input = &input[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while input.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&input[..64]);
            self.compress(&block);
            input = &input[64..];
        }
        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffered = input.len();
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        for (slot, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let length_bits = self.length_bits;
        self.update(&[0x80]);
        while self.buffered != 56 {
            self.update(&[0x00]);
        }
        self.length_bits = length_bits;
        self.update(&length_bits.to_be_bytes());
        let mut out = [0u8; 32];
        for (index, word) in self.state.iter().enumerate() {
            out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{CanonicalHasher, Sha256};

    fn hex(bytes: [u8; 32]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn sha256_matches_published_vectors() {
        let mut empty = Sha256::new();
        empty.update(b"");
        assert_eq!(
            hex(empty.finish()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let mut abc = Sha256::new();
        abc.update(b"abc");
        assert_eq!(
            hex(abc.finish()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let mut long = Sha256::new();
        long.update("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".as_bytes());
        assert_eq!(
            hex(long.finish()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_handles_a_multi_block_message() {
        let mut hasher = Sha256::new();
        for _ in 0..1000 {
            hasher.update(b"a");
        }
        assert_eq!(
            hex(hasher.finish()),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn canonical_encoding_is_injective_on_tags_and_lengths() {
        let mut left = CanonicalHasher::new();
        left.tag(1).str("ab").str("c");
        let mut right = CanonicalHasher::new();
        right.tag(1).str("a").str("bc");
        assert_ne!(left.finish(), right.finish());
    }
}
