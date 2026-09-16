//! Driver knowledge, the state domains it is grouped by, and the dirty set.
//!
//! Layer 2 exists to skip GL calls the driver does not need.  Skipping is only
//! sound while the mirror's belief about driver state is *known* rather than
//! assumed, so every mirrored value is a [`DriverKnowledge`] that starts
//! [`DriverKnowledge::Unknown`] and becomes `Known` only after the underlying
//! call succeeded.  There is deliberately no constructor that produces a
//! `Known` value from a GL default, because the defaults this layer would have
//! to assume are exactly the ones a Host, a debug layer, or a previous context
//! owner may already have changed.
//!
//! # Why a domain is a compile-time constant and not a string
//!
//! Domains are counted, reported and masked by index, so [`StateDomain`] is a
//! fieldless enum with an explicit [`StateDomain::COUNT`] and a stable order.
//! Adding a domain therefore changes the length of the counter array and the
//! width of [`DirtyDomains`] in one place, and the compiler refuses any match
//! that forgets it.

use core::fmt;

use super::cache::CacheMode;

/// One independently invalidatable group of driver state.
///
/// The grouping follows the plan's state model: a field belongs to the domain
/// that a *single* GL call or a single coherent call sequence establishes, so
/// that invalidating one domain never has to reason about another.  Where the
/// plan and the state-machine handoff overlap -- depth range is named in both
/// the session/output group and the program/pipeline group -- the value sits
/// with the pipeline, because GL sets it through the same rasterization state
/// the rest of that domain mirrors, and a session that owned it would have to
/// re-apply every pipeline's depth range at pass begin.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum StateDomain {
    /// Lifecycle, context stamp, active pass, draw/read framebuffer, draw
    /// buffers, render area, and the surface facts a pass renders through.
    Session,
    /// Program/pipeline identity, shader stage bindings, and the rasterization,
    /// depth, stencil and blend values a pipeline selects.
    Pipeline,
    /// Vertex-array identity, vertex slots and attributes, index binding, and
    /// the base-instance values that change attribute addressing.
    Geometry,
    /// Generic and indexed buffer bindings, including uniform and storage
    /// ranges, plus the pixel-store parameters an upload depends on.
    Buffers,
    /// Active texture unit, per-unit texture and sampler bindings, and storage
    /// image units.
    Textures,
    /// Common bind-group slots as logical identities plus their dynamic
    /// offsets, before they expand into the buffer and texture domains.
    Groups,
    /// Compute program identity, storage bindings, and pending memory
    /// visibility for profiles with the optional command domains.
    Compute,
    /// Active queries, sync objects, timer-query disjointness, and the bounded
    /// in-flight submission queue.
    Sync,
}

impl StateDomain {
    /// Every domain, in the order counters and masks are indexed by.
    pub(crate) const ALL: [Self; 8] = [
        Self::Session,
        Self::Pipeline,
        Self::Geometry,
        Self::Buffers,
        Self::Textures,
        Self::Groups,
        Self::Compute,
        Self::Sync,
    ];

    /// The number of domains, and therefore the width of a domain mask.
    pub(crate) const COUNT: usize = Self::ALL.len();

    /// This domain's index in [`StateDomain::ALL`].
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    /// The bit this domain occupies in [`DirtyDomains`].
    pub(crate) const fn bit(self) -> u32 {
        1 << self.index()
    }

    /// A short stable name for diagnostics and counter reports.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Pipeline => "pipeline",
            Self::Geometry => "geometry",
            Self::Buffers => "buffers",
            Self::Textures => "textures",
            Self::Groups => "groups",
            Self::Compute => "compute",
            Self::Sync => "sync",
        }
    }
}

impl fmt::Display for StateDomain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// A set of state domains, as a mask over [`StateDomain::ALL`].
///
/// The mask is a plain integer so that marking, clearing and testing a domain
/// cost no allocation on the submission path, which is the whole reason this
/// layer exists.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct DirtyDomains(u32);

impl DirtyDomains {
    /// No domain is dirty.
    pub(crate) const EMPTY: Self = Self(0);

    /// Every domain is dirty, which is the state after context creation,
    /// external invalidation, and context restoration.
    pub(crate) const ALL: Self = Self((1 << StateDomain::COUNT) - 1);

    /// A set holding exactly `domain`.
    pub(crate) const fn of(domain: StateDomain) -> Self {
        Self(domain.bit())
    }

    /// Adds `domain` to the set.
    pub(crate) fn insert(&mut self, domain: StateDomain) {
        self.0 |= domain.bit();
    }

    /// Removes `domain` from the set.
    pub(crate) fn remove(&mut self, domain: StateDomain) {
        self.0 &= !domain.bit();
    }

    /// Adds every domain in `domains`.
    pub(crate) fn extend(&mut self, domains: Self) {
        self.0 |= domains.0;
    }

    /// Returns whether `domain` is in the set.
    pub(crate) const fn contains(self, domain: StateDomain) -> bool {
        self.0 & domain.bit() != 0
    }

    /// Returns whether no domain is in the set.
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Removes every domain, returning the set that was there.
    pub(crate) fn take(&mut self) -> Self {
        core::mem::replace(self, Self::EMPTY)
    }

    /// The domains in this set, in [`StateDomain::ALL`] order.
    pub(crate) fn iter(self) -> impl Iterator<Item = StateDomain> {
        StateDomain::ALL
            .into_iter()
            .filter(move |domain| self.contains(*domain))
    }
}

impl fmt::Display for DirtyDomains {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("{")?;
        let mut first = true;
        for domain in self.iter() {
            if !first {
                formatter.write_str(", ")?;
            }
            first = false;
            formatter.write_str(domain.name())?;
        }
        formatter.write_str("}")
    }
}

/// What the mirror believes the driver currently holds for one value.
///
/// The two variants are the whole contract: `Unknown` means the next consumer
/// of this value must apply it, and `Known` means the last successful call left
/// exactly this value there.  Nothing in Layer 2 may compare a desired value
/// against an assumed default; that is what `Unknown` is for.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum DriverKnowledge<T> {
    /// The driver's value is not known and must be applied before use.
    Unknown,
    /// The driver accepted exactly this value.
    Known(T),
}

impl<T> DriverKnowledge<T> {
    /// The value the driver is known to hold, if any.
    pub(crate) const fn get(&self) -> Option<&T> {
        match self {
            Self::Unknown => None,
            Self::Known(value) => Some(value),
        }
    }

    /// Whether the driver's value is known.
    pub(crate) const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Records that the driver accepted `value`.
    pub(crate) fn set(&mut self, value: T) {
        *self = Self::Known(value);
    }

    /// Forgets the driver's value.
    pub(crate) fn invalidate(&mut self) {
        *self = Self::Unknown;
    }

    /// Whether the driver is known to hold exactly `desired`.
    ///
    /// This is the one predicate a redundant-call skip may be built on, and it
    /// is deliberately conservative in both directions: an `Unknown` value
    /// never agrees, and equality is the value's own `PartialEq` -- never a
    /// hash and never a structural shortcut a caller could get wrong.
    pub(crate) fn agrees(&self, desired: &T) -> bool
    where
        T: PartialEq,
    {
        match self {
            Self::Unknown => false,
            Self::Known(value) => value == desired,
        }
    }

    /// The value to hand a driver call: the known one, or `fallback`.
    ///
    /// Only for reads whose result cannot be avoided and whose fallback is
    /// itself correct -- a query this layer cannot skip.  It is named `or_else`
    /// rather than a `unwrap_or` so that a reader sees a fallback was chosen
    /// deliberately, and every call site has to justify the fallback it passes.
    pub(crate) fn or_else(self, fallback: T) -> T {
        match self {
            Self::Unknown => fallback,
            Self::Known(value) => value,
        }
    }
}

/// The two ways Layer 2 may execute a request.
///
/// The oracle is not a second production implementation: it is the same
/// domains, the same validation and the same call order with every skip
/// disabled, so that a differential test can compare an optimized trace against
/// the trace a machine with no mirror at all would have produced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExecutionMode {
    /// Skip every call the mirror proves redundant.
    Optimized,
    /// Emit every required call, keeping the mirror only for comparison.
    Oracle,
}

impl ExecutionMode {
    /// The cache policy this mode uses.
    pub(crate) const fn cache(self) -> CacheMode {
        match self {
            Self::Optimized => CacheMode::Enabled,
            Self::Oracle => CacheMode::Disabled,
        }
    }

    /// Whether a call proven redundant may be skipped.
    pub(crate) const fn may_skip(self) -> bool {
        matches!(self, Self::Optimized)
    }
}

#[cfg(test)]
mod tests {
    use super::{DirtyDomains, DriverKnowledge, ExecutionMode, StateDomain};
    use crate::webgl2::state::cache::CacheMode;

    #[test]
    fn every_domain_has_a_distinct_bit_and_index() {
        let mut seen = 0_u32;
        for (index, domain) in StateDomain::ALL.into_iter().enumerate() {
            assert_eq!(domain.index(), index);
            assert_eq!(seen & domain.bit(), 0, "{domain} reuses a bit");
            seen |= domain.bit();
        }
        assert_eq!(seen, DirtyDomains::ALL.0);
        assert_eq!(StateDomain::COUNT, 8);
    }

    #[test]
    fn unknown_never_agrees_and_known_agrees_by_value() {
        let mut blend = DriverKnowledge::Unknown;
        assert!(!blend.agrees(&true));
        assert!(!blend.agrees(&false));
        // An unknown value falls back rather than reporting a belief.
        assert!(!blend.or_else(false));

        blend.set(true);
        assert!(blend.agrees(&true));
        assert!(!blend.agrees(&false));
        assert!(blend.or_else(false));

        blend.invalidate();
        assert!(!blend.is_known());
        assert_eq!(blend.get(), None);
    }

    #[test]
    fn dirty_sets_report_and_clear_exactly_what_they_hold() {
        let mut dirty = DirtyDomains::EMPTY;
        assert!(dirty.is_empty());
        assert_eq!(dirty.to_string(), "{}");

        dirty.insert(StateDomain::Textures);
        dirty.extend(DirtyDomains::of(StateDomain::Sync));
        assert_eq!(dirty.to_string(), "{textures, sync}");
        assert!(dirty.contains(StateDomain::Textures));
        assert!(!dirty.contains(StateDomain::Compute));

        let taken = dirty.take();
        assert_eq!(taken.to_string(), "{textures, sync}");
        assert!(dirty.is_empty());
    }

    #[test]
    fn oracle_mode_disables_caching_and_skipping() {
        assert!(!ExecutionMode::Oracle.may_skip());
        assert_eq!(ExecutionMode::Oracle.cache(), CacheMode::Disabled);
        assert!(ExecutionMode::Optimized.may_skip());
        assert_eq!(ExecutionMode::Optimized.cache(), CacheMode::Enabled);
    }
}
