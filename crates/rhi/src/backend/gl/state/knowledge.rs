use core::fmt;

/// Independently invalidatable portions of a GL context.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum StateDomain {
    PassFramebuffer,
    RasterPipeline,
    Geometry,
    Bindings,
    Compute,
    PixelTransferCopy,
    Query,
    DerivedCaches,
}
impl StateDomain {
    pub(crate) const ALL: [Self; 8] = [
        Self::PassFramebuffer,
        Self::RasterPipeline,
        Self::Geometry,
        Self::Bindings,
        Self::Compute,
        Self::PixelTransferCopy,
        Self::Query,
        Self::DerivedCaches,
    ];
    pub(crate) const COUNT: usize = Self::ALL.len();
    pub(crate) const fn index(self) -> usize {
        self as usize
    }
    pub(crate) const fn bit(self) -> u32 {
        1 << self.index()
    }
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::PassFramebuffer => "pass/framebuffer",
            Self::RasterPipeline => "raster-pipeline",
            Self::Geometry => "geometry",
            Self::Bindings => "bindings",
            Self::Compute => "compute",
            Self::PixelTransferCopy => "pixel-transfer/copy",
            Self::Query => "query",
            Self::DerivedCaches => "derived-caches",
        }
    }
}
impl fmt::Display for StateDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Domains whose mirror values cannot be used for elision.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct DirtyDomains(u32);
impl DirtyDomains {
    pub(crate) const EMPTY: Self = Self(0);
    pub(crate) const ALL: Self = Self((1 << StateDomain::COUNT) - 1);
    pub(crate) const fn of(domain: StateDomain) -> Self {
        Self(domain.bit())
    }
    pub(crate) const fn contains(self, domain: StateDomain) -> bool {
        self.0 & domain.bit() != 0
    }
    pub(crate) fn insert(&mut self, domain: StateDomain) {
        self.0 |= domain.bit();
    }
    pub(crate) fn remove(&mut self, domain: StateDomain) {
        self.0 &= !domain.bit();
    }
    pub(crate) fn extend(&mut self, other: Self) {
        self.0 |= other.0;
    }
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub(crate) fn iter(self) -> impl Iterator<Item = StateDomain> {
        StateDomain::ALL
            .into_iter()
            .filter(move |d| self.contains(*d))
    }
}

/// A driver value is usable only after this state machine itself installed it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum DriverKnowledge<T> {
    Unknown,
    Known(T),
}
impl<T> DriverKnowledge<T> {
    pub(crate) const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }
    pub(crate) const fn get(&self) -> Option<&T> {
        match self {
            Self::Unknown => None,
            Self::Known(value) => Some(value),
        }
    }
    pub(crate) fn set(&mut self, value: T) {
        *self = Self::Known(value);
    }
    pub(crate) fn invalidate(&mut self) {
        *self = Self::Unknown;
    }
    pub(crate) fn agrees(&self, desired: &T) -> bool
    where
        T: PartialEq,
    {
        matches!(self, Self::Known(value) if value == desired)
    }
}

/// Differential-test mode.  Oracle emits every request, optimized emits only
/// requests whose known driver value differs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExecutionMode {
    Optimized,
    Oracle,
}
impl ExecutionMode {
    pub(crate) const fn may_skip(self) -> bool {
        matches!(self, Self::Optimized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_never_agrees() {
        let mut value = DriverKnowledge::Unknown;
        assert!(!value.agrees(&7));
        value.set(7);
        assert!(value.agrees(&7));
        value.invalidate();
        assert!(!value.agrees(&7));
    }
    #[test]
    fn domains_have_unique_bits() {
        let mut bits = 0;
        for d in StateDomain::ALL {
            assert_eq!(bits & d.bit(), 0);
            bits |= d.bit();
        }
        assert_eq!(bits, DirtyDomains::ALL.0);
    }
}
