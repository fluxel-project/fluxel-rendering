use crate::{FeatureHandle, RenderId};

/// The owner that supplied a Forward draw.
///
/// Scene records and feature-local draws deliberately occupy different
/// namespaces, even when their numeric components happen to match.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ForwardDrawSource {
    /// A draw selected from the immutable render-scene snapshot.
    Scene(RenderId),
    /// A draw supplied by one typed render feature.
    Feature {
        /// The feature that owns this draw.
        feature: FeatureHandle,
        /// Identity local to `feature` for this frame.
        local_id: u64,
    },
}

impl ForwardDrawSource {
    /// Returns the deterministic namespace-aware ordering key.
    pub(crate) fn sort_key(self) -> (u8, u64, u64) {
        match self {
            Self::Scene(record) => (0, u64::from(record.slot()), u64::from(record.generation())),
            Self::Feature { feature, local_id } => (1, feature.order_key(), local_id),
        }
    }
}

/// Ordering metadata attached to a prepared Forward draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForwardDrawOrder {
    source: ForwardDrawSource,
    order: i64,
}

impl ForwardDrawOrder {
    pub(crate) const fn new(source: ForwardDrawSource, order: i64) -> Self {
        Self { source, order }
    }

    /// Returns the source whose draw is ordered.
    pub const fn source(self) -> ForwardDrawSource {
        self.source
    }

    /// Returns the caller-provided ordering value.
    pub const fn order(self) -> i64 {
        self.order
    }

    /// Returns the deterministic opaque ordering key.
    ///
    /// The source portion is only a stable tie-break between otherwise equal
    /// ordering values; it does not transfer scheduling or Graph authority to
    /// a draw source.
    pub fn sort_key(self) -> (i64, (u8, u64, u64)) {
        (self.order, self.source.sort_key())
    }
}
