//! SHA-256, hand-written (specification section 7.1).
//!
//! # Why this exists rather than a `Hasher`
//!
//! [`crate::api::capability::CapabilityFingerprint`] is a 32-byte token that
//! tooling writes into a report and compares **across processes** — the
//! specification's own list of its uses is "cache key candidate, logs, artifact
//! provenance". `std::hash::DefaultHasher` cannot serve that:
//!
//! - its algorithm is not a stability guarantee, so a cache key built on it would
//!   silently stop matching when the toolchain is upgraded;
//! - it produces 64 bits, and padding that to 32 bytes would present a 64-bit
//!   digest as a 256-bit one.
//!
//! SHA-256 is stable by definition, standard, and checkable against the published
//! test vectors, which is why the tests below are transcriptions of them rather
//! than of this implementation's own output.
//!
//! # Why it is written here rather than depended on
//!
//! The crate carries no dependencies beyond the Windows bindings, and the
//! dependency audit is one of the conditions `version-plan.md` section 4 sets for
//! closing `0.16`. A digest is a closed specification, not a moving target: about
//! eighty lines that will never need to track a release.
//!
//! # What this is not
//!
//! Not a security boundary and not a collision guarantee. The specification is
//! explicit that equal fingerprints cannot alone carry correctness
//! (`01:1345-1346`), and the one token that *is* allowed to carry it —
//! [`crate::api::capability::CapabilityCompatibilityId`] — is interned by exact
//! byte comparison and never by a digest of any kind.

/// One SHA-256 compression round's constants: the first 32 bits of the fractional
/// part of the cube root of the first 64 primes.
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

/// The initial hash value: the first 32 bits of the fractional part of the square
/// root of the first 8 primes (FIPS 180-4 section 5.3.3).
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// An incremental SHA-256.
///
/// Incremental rather than one-shot because callers build their input from several
/// pieces — a canonical encoding walks a sorted collection and writes each element
/// — and buffering the whole encoding just to hash it would allocate for no
/// reason.
pub(crate) struct Sha256 {
    state: [u32; 8],
    /// Bytes not yet forming a complete 64-byte block.
    pending: [u8; 64],
    pending_len: usize,
    /// Total message length in bytes, for the padding's length field.
    ///
    /// Bytes, not bits: the shift to bits happens once, at padding time, where the
    /// conversion is visible. Counting in bits here would need `u64` shifts on
    /// every update for no benefit.
    total: u64,
}

impl Sha256 {
    /// Starts a digest.
    pub(crate) fn new() -> Self {
        Self {
            state: H0,
            pending: [0; 64],
            pending_len: 0,
            total: 0,
        }
    }

    /// Absorbs `bytes`.
    pub(crate) fn update(&mut self, bytes: &[u8]) {
        self.total = self.total.wrapping_add(bytes.len() as u64);
        let mut rest = bytes;
        if self.pending_len > 0 {
            let want = 64 - self.pending_len;
            let take = want.min(rest.len());
            self.pending[self.pending_len..self.pending_len + take].copy_from_slice(&rest[..take]);
            self.pending_len += take;
            rest = &rest[take..];
            if self.pending_len == 64 {
                let block = self.pending;
                self.compress(&block);
                self.pending_len = 0;
            }
        }
        let mut chunks = rest.chunks_exact(64);
        for block in &mut chunks {
            let mut block64 = [0u8; 64];
            block64.copy_from_slice(block);
            self.compress(&block64);
        }
        // Only a *drained* pending block can be refilled from index zero. When the
        // partial fill above ran out of input before reaching 64 bytes, `rest` is
        // empty here, and treating the empty remainder as "the new pending block"
        // would reset `pending_len` to zero and silently discard the bytes already
        // buffered — a wrong digest, not an error. A non-empty remainder implies
        // the drain happened, which is what the assertion records.
        let tail = chunks.remainder();
        if !tail.is_empty() {
            debug_assert_eq!(
                self.pending_len, 0,
                "a non-empty remainder means the pending block was drained"
            );
            self.pending[..tail.len()].copy_from_slice(tail);
            self.pending_len = tail.len();
        }
    }

    /// Finishes the digest and returns the 32-byte result.
    pub(crate) fn finish(mut self) -> [u8; 32] {
        // Padding (FIPS 180-4 section 5.1.1): a single 1 bit, then zeros, then the
        // message length in *bits* as a big-endian u64. The length goes in the last
        // 8 bytes of a block, so the zero fill runs to byte 56 of the final block —
        // or to byte 56 of the *next* one when the current block has fewer than 56
        // bytes left.
        let bit_len = self.total.wrapping_mul(8);
        self.pending[self.pending_len] = 0x80;
        self.pending_len += 1;
        if self.pending_len > 56 {
            for byte in &mut self.pending[self.pending_len..] {
                *byte = 0;
            }
            let block = self.pending;
            self.compress(&block);
            self.pending_len = 0;
        }
        for byte in &mut self.pending[self.pending_len..56] {
            *byte = 0;
        }
        self.pending[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.pending;
        self.compress(&block);

        let mut out = [0u8; 32];
        for (index, word) in self.state.iter().enumerate() {
            out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// One compression function application.
    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (index, word) in w.iter_mut().take(16).enumerate() {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&block[index * 4..index * 4 + 4]);
            *word = u32::from_be_bytes(bytes);
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
}

/// Digests `bytes` in one call.
pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finish()
}

#[cfg(test)]
mod tests;
