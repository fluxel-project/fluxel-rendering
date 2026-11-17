//! Public policy for bounded dual-depth-peeling Forward draws.

use std::num::NonZeroU32;

use crate::ForwardDrawSource;

/// Selection policy for the renderer's dual-depth-peeling Forward path.
///
/// This only selects the number of peel rounds and transparent sources kept
/// out of the peel segment. It never substitutes sorted rendering when OIT
/// capability admission or its actual resources are unavailable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DualDepthPeelingConfig {
    rounds: NonZeroU32,
    excluded_sources: Vec<ForwardDrawSource>,
}

impl DualDepthPeelingConfig {
    /// Babylon-compatible default number of peel rounds.
    pub const DEFAULT_ROUNDS: NonZeroU32 = match NonZeroU32::new(5) {
        Some(rounds) => rounds,
        None => unreachable!(),
    };

    /// Creates a bounded depth-peeling policy with no excluded sources.
    #[must_use]
    pub fn new(rounds: NonZeroU32) -> Self {
        Self {
            rounds,
            excluded_sources: Vec::new(),
        }
    }

    /// Returns the configured number of dual-depth-peeling rounds.
    #[must_use]
    pub const fn rounds(&self) -> NonZeroU32 {
        self.rounds
    }

    /// Keeps one transparent source out of the OIT peel segment.
    ///
    /// Repeating a source is harmless and does not change its partition.
    #[must_use]
    pub fn with_excluded_source(mut self, source: ForwardDrawSource) -> Self {
        if !self.excludes_source(source) {
            self.excluded_sources.push(source);
        }
        self
    }

    /// Returns whether `source` is excluded from the OIT peel segment.
    #[must_use]
    pub fn excludes_source(&self, source: ForwardDrawSource) -> bool {
        self.excluded_sources.contains(&source)
    }

    /// Iterates the explicitly excluded transparent draw sources.
    pub fn excluded_sources(&self) -> impl ExactSizeIterator<Item = ForwardDrawSource> + '_ {
        self.excluded_sources.iter().copied()
    }
}

impl Default for DualDepthPeelingConfig {
    fn default() -> Self {
        Self::new(Self::DEFAULT_ROUNDS)
    }
}
