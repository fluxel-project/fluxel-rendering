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
    TimerQuery,
}
impl GlCapability {
    const fn requires_probe(self) -> bool {
        !matches!(self, Self::TimerQuery)
    }
    fn limits_satisfied(self, l: &GlLimits, formats: &GlFormatTable) -> bool {
        match self {
            Self::Compute => l.supports_compute(),
            Self::StorageBuffer => l.supports_storage_buffers(),
            Self::StorageImage => l.supports_storage_images() && formats.has_storage_read_write(),
            Self::IndirectDraw | Self::IndirectDispatch => l.supports_single_indirect(),
            Self::MultiDrawIndirect => l.supports_multi_draw_indirect(),
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

/// The only construction path for a discovery snapshot.
#[derive(Debug)]
pub(crate) struct GlDiscoveryBuilder {
    stamp: ContextStamp,
    context: GlContextInfo,
    extensions: GlExtensionSet,
    limits: GlLimits,
    formats: GlFormatTable,
    capabilities: GlCapabilitySet,
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
        })
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
