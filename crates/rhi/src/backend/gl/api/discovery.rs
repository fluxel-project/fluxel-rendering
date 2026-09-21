//! Durable, context-generation-bound GL-family discovery snapshots.

use super::{
    CapabilityEvidence, ContextStamp, CoreOrExtension, GlExtensionSet, GlFamilyProfile,
    GlFormatTable, GlLimits,
};
use std::collections::{BTreeMap, BTreeSet};

/// Exact context identity strings, retained without lossy parsing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlContextInfo {
    profile: GlFamilyProfile,
    version: String,
    shading_language_version: String,
    vendor: String,
    renderer: String,
    driver_or_browser: String,
    flags: GlContextFlags,
}
impl GlContextInfo {
    /// Creates the identity from exact provider strings.
    pub(crate) fn new(
        profile: GlFamilyProfile,
        version: impl Into<String>,
        shading_language_version: impl Into<String>,
        vendor: impl Into<String>,
        renderer: impl Into<String>,
        driver_or_browser: impl Into<String>,
        flags: GlContextFlags,
    ) -> Self {
        Self {
            profile,
            version: version.into(),
            shading_language_version: shading_language_version.into(),
            vendor: vendor.into(),
            renderer: renderer.into(),
            driver_or_browser: driver_or_browser.into(),
            flags,
        }
    }
    /// Returns the selected GL-family profile.
    pub(crate) const fn profile(&self) -> GlFamilyProfile {
        self.profile
    }
    /// Returns the raw GL/WebGL version.
    pub(crate) fn version(&self) -> &str {
        &self.version
    }
    /// Returns the raw GLSL version.
    pub(crate) fn shading_language_version(&self) -> &str {
        &self.shading_language_version
    }
    /// Returns the vendor string.
    pub(crate) fn vendor(&self) -> &str {
        &self.vendor
    }
    /// Returns the renderer string.
    pub(crate) fn renderer(&self) -> &str {
        &self.renderer
    }
    /// Returns driver or browser identity.
    pub(crate) fn driver_or_browser(&self) -> &str {
        &self.driver_or_browser
    }
    /// Returns context flags.
    pub(crate) fn flags(&self) -> &GlContextFlags {
        &self.flags
    }
}

/// Normalized and provider-specific context flags.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GlContextFlags {
    pub debug: bool,
    pub forward_compatible: bool,
    pub robust_access: bool,
    pub no_error: bool,
    pub other: BTreeSet<String>,
}

/// Common semantics whose evidence must be recorded.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlCapability {
    Compute,
    StorageBuffer,
    StorageImage,
    IndirectDraw,
    IndirectDispatch,
    MultiDrawIndirect,
    /// One command issuing many draws from per-draw parameter slices.
    MultiDraw,
    /// One attachment serving several array layers in a single pass.
    Multiview,
    TimerQuery,
}
impl GlCapability {
    const fn requires_probe(self) -> bool {
        match self {
            Self::TimerQuery => false,
            // A per-draw parameter batch is proved by its acquired, complete
            // and callable command set: the domain has no queryable limit and
            // discovery never installs the pipeline a scratch draw would need,
            // so the entry-point oracle is the strongest evidence available
            // before the first real draw.
            Self::MultiDraw => false,
            _ => true,
        }
    }
    fn limits_satisfied(self, l: &GlLimits, formats: &GlFormatTable) -> bool {
        match self {
            Self::Compute => l.supports_compute(),
            Self::StorageBuffer => l.supports_storage_buffers(),
            Self::StorageImage => l.supports_storage_images() && formats.has_storage_read_write(),
            Self::IndirectDraw | Self::IndirectDispatch => l.supports_single_indirect(),
            Self::MultiDrawIndirect => l.supports_multi_draw_indirect(),
            // Every draw of a batch is validated exactly like the single-draw
            // path, so this domain adds no numeric requirement of its own and
            // an unconditionally satisfied limit half is the honest fact.
            Self::MultiDraw => true,
            Self::Multiview => l.supports_multiview(),
            Self::TimerQuery => l.query_counter_bits != 0,
        }
    }
}

/// Result of an actual command-domain probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlOperationProbe {
    NotRequired,
    Passed,
    Failed,
    NotRun,
}
/// Durable evidence for one normalized capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlCapabilityFact {
    pub evidence: Option<CapabilityEvidence>,
    pub limits_satisfied: bool,
    pub operation_probe: GlOperationProbe,
}
impl GlCapabilityFact {
    const fn is_enabled(self, c: GlCapability) -> bool {
        self.evidence.is_some()
            && self.limits_satisfied
            && (!c.requires_probe() || matches!(self.operation_probe, GlOperationProbe::Passed))
    }
}
/// Immutable per-context capability ledger.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GlCapabilitySet {
    facts: BTreeMap<GlCapability, GlCapabilityFact>,
}
impl GlCapabilitySet {
    pub(crate) fn fact(&self, c: GlCapability) -> Option<GlCapabilityFact> {
        self.facts.get(&c).copied()
    }
    pub(crate) fn supports(&self, c: GlCapability) -> bool {
        self.fact(c).is_some_and(|fact| fact.is_enabled(c))
    }
}

/// The drawable behind this context, as the one fact a presenter acts on.
///
/// FBO 0 has no `GlFormatTable` row, so the surface the platform flips is the one
/// piece of format evidence no other record carries, and a presenter needs it to
/// know what it is presenting.  The `gl.surface-*` keys that report it are a
/// formatted channel; this is the acting one, and it exists because the consumer
/// that arrived -- presentation, which must name the formats a graph may compile
/// a present root against -- needs component widths as numbers, and would
/// otherwise be parsing a string it wrote itself.
///
/// The keys stay, and both renderings come from one observation rather than one
/// being read back out of the other: a reader already depends on the keys, and a
/// key/value pair cannot become a struct field without breaking that reader.  The
/// *reason* an observation failed stays with the keys for the same reason -- this
/// value answers "can a presenter claim a format here", and the answer is the
/// same for every way it cannot.
///
/// It holds the component widths and nothing else, because they are what the
/// claim is made of: the drawable's depth, stencil and sample counts are observed
/// and stay in the keys, and no field of the common contract's surface row is
/// derived from them.  A field nothing reads is not a fact this value carries, it
/// is a fact this value would have to keep agreeing with the keys about.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum GlSurfaceFacts {
    /// Widths read from the bound draw framebuffer, in RGBA order.
    Observed {
        /// Bits per colour component, in RGBA order.
        color_bits: [u32; 4],
    },
    /// Nothing was observed, so no format may be claimed.
    ///
    /// The default, and deliberately so: a builder that was never told what the
    /// drawable is reports no surface, which is the answer that rejects work.
    #[default]
    Unavailable,
}

/// The only construction path for a discovery snapshot.
#[derive(Debug)]
pub(crate) struct GlDiscoveryBuilder {
    stamp: ContextStamp,
    context: GlContextInfo,
    extensions: GlExtensionSet,
    limits: GlLimits,
    formats: GlFormatTable,
    capabilities: GlCapabilitySet,
    surface_facts: GlSurfaceFacts,
}
impl GlDiscoveryBuilder {
    /// Binds all raw observations to one exact context generation before resolution.
    pub(crate) fn new(
        stamp: ContextStamp,
        context: GlContextInfo,
        extensions: GlExtensionSet,
        limits: GlLimits,
        formats: GlFormatTable,
    ) -> Result<Self, GlDiscoveryError> {
        validate_profile(context.profile())?;
        limits
            .validate_profile_minimums(context.profile())
            .map_err(GlDiscoveryError::BelowProfileMinimum)?;
        formats
            .validate_for_limits(&limits)
            .map_err(GlDiscoveryError::InvalidFormats)?;
        formats
            .validate_evidence(context.profile(), &extensions)
            .map_err(GlDiscoveryError::InvalidFormats)?;
        Ok(Self {
            stamp,
            context,
            extensions,
            limits,
            formats,
            capabilities: GlCapabilitySet::default(),
            surface_facts: GlSurfaceFacts::Unavailable,
        })
    }
    /// Binds what the provider observed about the drawable.
    ///
    /// Separate from [`Self::new`] because the drawable is not part of what makes
    /// a builder constructible -- a context is discovered before anything asks
    /// what its default framebuffer looks like -- and because leaving it unset
    /// has to keep meaning what it means today: no surface is claimed.
    pub(crate) fn surface_facts(&mut self, facts: GlSurfaceFacts) {
        self.surface_facts = facts;
    }
    /// Resolves capability evidence exclusively from this builder's bound context and extensions.
    pub(crate) fn resolve(
        &mut self,
        capability: GlCapability,
        requirement: CoreOrExtension,
        operation_probe: GlOperationProbe,
    ) {
        let fact = GlCapabilityFact {
            evidence: requirement.resolve(self.context.profile(), &self.extensions),
            limits_satisfied: capability.limits_satisfied(&self.limits, &self.formats),
            operation_probe,
        };
        self.capabilities.facts.insert(capability, fact);
    }
    /// Freezes all observations and internally resolved capability facts.
    pub(crate) fn build(self) -> GlDiscoverySnapshot {
        GlDiscoverySnapshot {
            stamp: self.stamp,
            context: self.context,
            extensions: self.extensions,
            limits: self.limits,
            formats: self.formats,
            capabilities: self.capabilities,
            surface_facts: self.surface_facts,
        }
    }
}

/// Complete immutable evidence for exactly one context epoch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlDiscoverySnapshot {
    stamp: ContextStamp,
    context: GlContextInfo,
    extensions: GlExtensionSet,
    limits: GlLimits,
    formats: GlFormatTable,
    capabilities: GlCapabilitySet,
    surface_facts: GlSurfaceFacts,
}
impl GlDiscoverySnapshot {
    /// Returns the exact context generation that authorized this snapshot.
    pub(crate) const fn context_stamp(&self) -> ContextStamp {
        self.stamp
    }
    /// Returns identity facts.
    pub(crate) fn context(&self) -> &GlContextInfo {
        &self.context
    }
    /// Returns raw and typed extension evidence.
    pub(crate) fn extensions(&self) -> &GlExtensionSet {
        &self.extensions
    }
    /// Returns numerical facts.
    pub(crate) const fn limits(&self) -> GlLimits {
        self.limits
    }
    /// Returns exact format/count facts.
    pub(crate) fn formats(&self) -> &GlFormatTable {
        &self.formats
    }
    /// Returns normalized capability evidence.
    pub(crate) fn capabilities(&self) -> &GlCapabilitySet {
        &self.capabilities
    }
    /// Returns what the provider observed about the drawable.
    ///
    /// The typed reading of the `gl.surface-*` context flags, recorded from the
    /// same observation.  See [`GlSurfaceFacts`] for why both exist.
    pub(crate) fn surface_facts(&self) -> GlSurfaceFacts {
        self.surface_facts
    }

    /// Returns how many views one attachment of a pass may serve here.
    ///
    /// A context that did not prove multiview reports the single view that
    /// every pass already uses, which is the WebGPU default for
    /// `maxMultiviewViewCount`; a queried view count is never reported for a
    /// capability that did not enable, so a permissive number can never leak
    /// past a failed capability fact.
    pub(crate) fn max_multiview_view_count(&self) -> u32 {
        if self.capabilities.supports(GlCapability::Multiview) {
            self.limits.max_multiview_view_count
        } else {
            1
        }
    }

    /// Rebinds immutable observations for deterministic context-restore tests.
    ///
    /// Real providers must rediscover after restoration and cannot call this
    /// test-only seam.
    #[cfg(test)]
    pub(super) fn rebind_for_test(&self, stamp: ContextStamp) -> Self {
        let mut rebound = self.clone();
        rebound.stamp = stamp;
        rebound
    }
}

/// Discovery construction failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlDiscoveryError {
    InvalidProfile(GlFamilyProfile),
    BelowProfileMinimum(super::GlLimitViolation),
    InvalidFormats(super::GlFormatTableError),
}
fn validate_profile(profile: GlFamilyProfile) -> Result<(), GlDiscoveryError> {
    match profile {
        GlFamilyProfile::Desktop { major: 4, .. }
        | GlFamilyProfile::Embedded { major: 3, .. }
        | GlFamilyProfile::WebGl2 => Ok(()),
        _ => Err(GlDiscoveryError::InvalidProfile(profile)),
    }
}
